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

#%% Load dataset and cuda
dataset = pd.read_csv("datasets/mini-dataset.csv")
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
protein = dataset.loc[:, "uniprotID"]    # 5
ligand = dataset.loc[:, "chemblID"]
kiba = list(dataset['KIBA'])
# del dataset

#%% protein sequence load
f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
seq_voc, _ = pickle.load(f)
f.close()

sequence = np.zeros(((2**13)+(2**8), 4128))
for i, s in enumerate(protein):
    sequence[i] = seq_voc[s]

sequence = sequence[:, :2000]

#%% ligand ecfp load
f = open('datasets/dictionaries/lgn_ecfp.txt', 'rb')
ecfp = pickle.load(f)
f.close()

ecfprint = np.zeros(((2**13)+(2**8), 2048))
for i, e in enumerate(ligand):
    ecfprint[i] = ecfp[e]

#%% dataset zip
revised_dataset = list(zip(sequence, ecfprint, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

del shuffled_dataset
    
#%% Make collate func.
def collate(samples):
    sequences, ecfprints, labels = map(list, zip(*samples))
    return tc.LongTensor(sequences).cuda(), tc.tensor(ecfprints, dtype=tc.float).cuda(), tc.tensor(labels).cuda()

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.emlayer = nn.Embedding(21, 10)
        
        self.cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 8, kernel_size = (64, 10), stride=(2, 1)),
                        nn.BatchNorm2d(num_features = 8),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = (2, 1))
                        )
        
        self.cv1dlayers = nn.Sequential(
                        nn.Conv1d(8, 16, kernel_size = 3),
                        nn.BatchNorm1d(num_features = 16),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 2),
                        nn.Conv1d(16, 24, kernel_size = 2),
                        nn.BatchNorm1d(num_features = 24),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 2)
                        )
            
        self.lslayer = nn.LSTM(24, 64, num_layers=1, bidirectional=True, batch_first=True)
        
        self.eclayers = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 128),
                        )

        self.regress = nn.Linear(256, 1, F.elu)    # regression

    def forward(self, seq, ecfp):        
        cv_i = self.emlayer(seq)
        
        cv2_i = cv_i.unsqueeze(1)
        cv2_o = self.cv2dlayer(cv2_i)
        cv1_i = cv2_o.squeeze()
        cv1_o = self.cv1dlayers(cv1_i)
        
        ls_i = cv1_o.permute(0, 2, 1)
        
        ls_h = torch.zeros(2, 32, 64).cuda()     # (num_layers * num_directions, batch, hidden_size)
        ls_c = torch.zeros(2, 32, 64).cuda()
        
        ls_o, (ls_h, ls_c) = self.lslayer(ls_i, (ls_h, ls_c))

        for_o = ls_o[:, -1, :64]
        back_o = ls_o[:, 0, 64:]
        
        concat_o = tc.cat((for_o, back_o), axis=1)   # batch, hidden*2
        
        ec_h = self.eclayers(ecfp)
        dim = 1
        for e in ec_h.size()[1:]:
            dim = dim * e
        ec_h = ec_h.view(-1, dim)
        
        cat = tc.cat((concat_o, ec_h), axis=1).cuda()
       
        return self.regress(cat).cuda()

#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 32
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

    for iter, (seq, ecfp, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(seq, ecfp).view(-1).cuda()
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

    for iter, (seq, ecfp, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(seq, ecfp).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)

#%%
np.save('cnnlstm+ecfp_tr_losses', tr_epoch_losses)
np.save('cnnlstm+ecfp_va_losses', va_epoch_losses)
