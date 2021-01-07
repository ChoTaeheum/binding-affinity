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
protein = dataset.loc[:(2**13)+(2**8)-1, "uniprotID"]    # 5
ligand = dataset.loc[:(2**13)+(2**8)-1, "chemblID"]
kiba = list(dataset.loc[:(2**13)+(2**8)-1, 'KIBA'])
del dataset

#%% protein sequence load
f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
seq_voc, _ = pickle.load(f)
f.close()

sequence = np.zeros(((2**13)+(2**8), 4128))
for i, s in enumerate(protein):
    sequence[i] = seq_voc[s]

sequence = sequence[:, :1000]


#%% ligand ecfp load
f = open('datasets/dictionaries/lgn_smiecoding.txt', 'rb')
smi_dic = pickle.load(f)
f.close()

smileseq = np.zeros(((2**13)+(2**8), 590))
for i, e in enumerate(ligand):
    smileseq[i] = smi_dic[e]
    
smileseq = smileseq[:, :200]

#%% dataset zip
revised_dataset = list(zip(sequence, smileseq, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

del shuffled_dataset


#%% Make collate func.
def collate(samples):
    sequences, smileseq, labels = map(list, zip(*samples))
    return tc.LongTensor(sequences).cuda(), tc.LongTensor(smileseq).cuda(), tc.tensor(labels).cuda()

#%%
class Conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(Conv1d, self).__init__()
        
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=9, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)

    def forward(self, x):
        x = self.conv(x)
        out = self.bn(x)
        return out
    
    
class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        
        self.conv = nn.Sequential(
            Conv1d(32, 48, kernel_size=10, stride=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2)            
            )
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
    
class IBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(IBlock, self).__init__()
        
        self.branch0 = Conv1d(in_planes, out_planes, kernel_size=9, stride=1)
        
        self.branch1 = nn.Sequential(
            Conv1d(in_planes, in_planes, kernel_size=9, stride=1),
            nn.ReLU(),
            Conv1d(in_planes, in_planes, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            Conv1d(in_planes, out_planes, kernel_size=9, stride=1, padding=4)
            )
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = x0 + x1
        out = self.relu(x2)
        
        return out
            


#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.prt_emlayer = nn.Embedding(21, 10)
        
        self.prt_cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = (4, 10)),
                        nn.BatchNorm2d(num_features = 32),
                        nn.ReLU()    # batch, channel, input_len, embedding
                        )    
        
        self.prt_cv1dlayers = nn.Sequential(
                        Block1(),
                        IBlock(48, 48),
                        IBlock(48, 192),
                        nn.MaxPool1d(4),
                        IBlock(192, 96),
                        IBlock(96, 384),
                        nn.MaxPool1d(4),
                        IBlock(384, 1536),
                        nn.AvgPool1d(17)
                        )
            
        ######################################################################
        ######################################################################
        
        self.lgn_emlayer = nn.Embedding(64, 10)
        
        self.lgn_cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = (2, 10)),
                        nn.BatchNorm2d(num_features = 32),
                        nn.ReLU()   
                        )
        
        self.lgn_cv1dlayers = nn.Sequential(
                        Block1(),
                        IBlock(48, 48),
                        IBlock(48, 192),
                        IBlock(192, 96),
                        nn.MaxPool1d(2),
                        IBlock(96, 384),
                        IBlock(384, 1536),
                        nn.AvgPool1d(19)
                        )
            
        
        self.mlplayers = nn.Sequential(
                        nn.Linear(3072, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU()
                        )

        self.regress = nn.Linear(512, 1)    # regression

    def forward(self, prt_seq, lgn_seq):   
        p = self.prt_emlayer(prt_seq)
        p = p.unsqueeze(1)
        p = self.prt_cv2dlayer(p)
        p = p.squeeze()
        p = self.prt_cv1dlayers(p)     # batch, channel(->input_size), seq_len
        p = p.squeeze()
        
        l = self.lgn_emlayer(lgn_seq)
        l = l.unsqueeze(1)
        l = self.lgn_cv2dlayer(l)
        l = l.squeeze()
        l = self.lgn_cv1dlayers(l)
        l = l.squeeze()
        
        cat = tc.cat((p, l), axis=1).cuda()
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
np.save('ModifiedDeepDTA_v4_tr_losses', tr_epoch_losses)
np.save('ModifiedDeepDTA_v4_va_losses', va_epoch_losses)
min(va_epoch_losses)