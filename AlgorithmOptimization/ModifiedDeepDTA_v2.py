#%% library import
import numpy as np
import pandas as pd
import networkx as nx
import torch as tc
import torch
import pprint
import pickle
import time

from torch.autograd import Variable
from sklearn.utils import shuffle
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

###########################################################################################
#%% Load dataset and cuda
dataset = pd.read_csv("datasets/KIBA.csv")
datalen = len(dataset)
cuda = tc.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

#%% protein-ligand-kiba split
protein = dataset.loc[:(2**16)+(2**13)-1, "uniprotID"]    # 5
ligand = dataset.loc[:(2**16)+(2**13)-1, "chemblID"]
kiba = list(dataset.loc[:(2**16)+(2**13)-1, 'KIBA'])
del dataset

#%% protein sequence load
f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
seq_voc, _ = pickle.load(f)
f.close()

sequence = np.zeros(((2**16)+(2**13), 4128))
for i, s in enumerate(protein):
    sequence[i] = seq_voc[s]

sequence = sequence[:, :2000]

#%% ligand ecfp load
f = open('datasets/dictionaries/lgn_smiecoding.txt', 'rb')
smi_dic = pickle.load(f)
f.close()

smileseq = np.zeros(((2**16)+(2**13), 590))
for i, e in enumerate(ligand):
    smileseq[i] = smi_dic[e]
    
smileseq = smileseq[:, :100]

#%% dataset zip
revised_dataset = list(zip(sequence, smileseq, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**16]
validset = shuffled_dataset[2**16:(2**16) + (2**13)]

del shuffled_dataset

#%% Make collate func.
def collate(samples):
    sequences, smileseq, labels = map(list, zip(*samples))
    return tc.LongTensor(sequences).cuda(), tc.LongTensor(smileseq).cuda(), tc.tensor(labels).cuda()

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.prt_emlayer = nn.Embedding(21, 10)
        
        self.prt_cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = (4, 10)),
                        nn.BatchNorm2d(num_features = 32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = (2,1))
                        )
        
        self.prt_cv1dlayers = nn.Sequential(
                        nn.Conv1d(32, 64, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 2),
                        nn.Conv1d(64, 96, kernel_size = 12),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 4)
                        )
            
        self.lslayer = nn.LSTM(96, 64, num_layers=1, bidirectional=True, batch_first=True)
        
        ######################################################################
        ######################################################################
        
        self.lgn_emlayer = nn.Embedding(64, 10)
        
        self.lgn_cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = (4, 10)),
                        nn.BatchNorm2d(num_features = 32),
                        nn.ReLU()
                        )
        
        self.lgn_cv1dlayers = nn.Sequential(
                        nn.Conv1d(32, 64, kernel_size = 6),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.Conv1d(64, 96, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU()
                        )
            
        self.lslayer = nn.LSTM(96, 64, num_layers=1, bidirectional=True, batch_first=True)    # input_size, hidden_size
        
        self.mlplayers = nn.Sequential(
                        nn.Linear(256, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                        )

        self.regress = nn.Linear(512, 1)    # regression

    def forward(self, prt_seq, lgn_seq):   
        p = self.prt_emlayer(prt_seq)
        p = p.unsqueeze(1)
        p = self.prt_cv2dlayer(p)
        p = p.squeeze()
        p = self.prt_cv1dlayers(p)     # batch, channel(->input_size), seq_len
        p = p.permute(0, 2, 1)
        
        p_h = torch.zeros(2, 128, 64).cuda()     # (num_layers * num_directions, batch, hidden_size)
        p_c = torch.zeros(2, 128, 64).cuda()
        p_o, (p_h, p_c) = self.lslayer(p, (p_h, p_c))
        p_fo = p_o[:, -1, :64]; p_bo = p_o[:, 0, 64:]
        
        
        l = self.lgn_emlayer(lgn_seq)
        l = l.unsqueeze(1)
        l = self.lgn_cv2dlayer(l)
        l = l.squeeze()
        l = self.lgn_cv1dlayers(l)
        l = l.permute(0, 2, 1)
        
        l_h = torch.zeros(2, 128, 64).cuda()     # (num_layers * num_directions, batch, hidden_size)
        l_c = torch.zeros(2, 128, 64).cuda()
        l_o, (l_h, l_c) = self.lslayer(l, (l_h, l_c))
        l_fo = l_o[:, -1, :64]; l_bo = l_o[:, 0, 64:]
        
        
        cat = tc.cat((p_fo, p_bo, l_fo, l_bo), axis=1).cuda()
        out = self.mlplayers(cat)
        
        return self.regress(out).cuda()

#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 128
hp_d['num_epochs'] = 300

hp_d['init_learning_rate'] = 10 ** -3.70183
hp_d['eps'] = 10 ** -8.39981
hp_d['weight_decay'] = 10 ** -3.59967

#%% training and validation
tr_data_loader = DataLoader(trainset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)
va_data_loader = DataLoader(validset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)

model = Regressor().to(torch.device('cuda:0'))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
loss_func = nn.MSELoss(reduction='mean').cuda()
optimizer = optim.Adam(model.parameters(), lr=hp_d['init_learning_rate'], 
    weight_decay=hp_d['weight_decay'], eps=hp_d['eps'])

print('tr_var:', np.var(np.array([s[2] for s in trainset])))
print('va_var:', np.var(np.array([s[2] for s in validset])))
print('total params:', total_params)

tr_epoch_losses = []
va_epoch_losses = []

start = time.time()

for epoch in range(hp_d['num_epochs']):                          #!! epoch-loop
    # training session
    model.train()
    tr_epoch_loss = 0

    for iter, (seq, smi, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(seq, smi).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tr_epoch_loss += loss.detach().item()
    
    tr_epoch_loss /= (iter + 1)
    print('Training epoch {}, loss {:.4f}'.format(epoch, tr_epoch_loss))
    tr_epoch_losses.append(tr_epoch_loss)

# ===========================================================================
    # validation session
    model.eval()
    va_epoch_loss = 0

    for iter, (seq, smi, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(seq, smi).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)

#%%
np.save('ModifiedDeepDTA_v2_tr_losses', tr_epoch_losses)
np.save('ModifiedDeepDTA_v2_va_losses', va_epoch_losses)
