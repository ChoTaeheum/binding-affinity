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
from sklearn.model_selection import KFold

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
protein = dataset.loc[:, "uniprotID"]    # 5
ligand = dataset.loc[:, "chemblID"]
kiba = list(dataset.loc[:, 'KIBA'])
del dataset

#%% protein sequence load
f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
seq_voc, _ = pickle.load(f)
f.close()

sequence = np.zeros((datalen, 4128))
for i, s in enumerate(protein):
    sequence[i] = seq_voc[s]

sequence = sequence[:, :1400]

#%% ligand smiles load
f = open('datasets/dictionaries/lgn_smiecoding.txt', 'rb')
smi_dic = pickle.load(f)
f.close()

smileseq = np.zeros((datalen, 590))
for i, e in enumerate(ligand):
    smileseq[i] = smi_dic[e]

smileseq = smileseq[:, :100]

#%% dataset zip
revised_dataset = list(zip(sequence, smileseq, kiba))
shuffled_dataset = np.array(shuffle(revised_dataset)); del revised_dataset

#%% Make collate func.
def collate(samples):
    sequences, smileseq, labels = map(list, zip(*samples))
    return tc.LongTensor(sequences).cuda(), tc.LongTensor(smileseq).cuda(), tc.tensor(labels).cuda()

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.prt_emlayer = nn.Embedding(21, 10)
        
        self.prt_cv1dlayers = nn.Sequential(
                        nn.Conv1d(10, 32, kernel_size = 4),
                        nn.BatchNorm1d(num_features = 32),
                        nn.ReLU(),
                        nn.Conv1d(32, 64, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.Conv1d(64, 96, kernel_size = 12),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=1379)
                        )
        
        
        ######################################################################
        ######################################################################
        
        self.lgn_emlayer = nn.Embedding(64, 10)
        
        self.lgn_cv1dlayers = nn.Sequential(
                        nn.Conv1d(10, 32, kernel_size = 4),
                        nn.BatchNorm1d(num_features = 32),
                        nn.ReLU(),
                        nn.Conv1d(32, 64, kernel_size = 6),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.Conv1d(64, 96, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 85)
                        )
        
        
        self.mlplayers = nn.Sequential(
                        nn.Linear(192, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
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
                        nn.Dropout(0.1),
                        nn.Linear(512, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU()
                        )

        self.regress = nn.Linear(512, 1)    # regression

    def forward(self, prt_seq, lgn_seq):   
        p = self.prt_emlayer(prt_seq)
        p = p.permute(0, 2, 1)
        p = self.prt_cv1dlayers(p)
        p = p.squeeze()
        
        l = self.lgn_emlayer(lgn_seq)
        l = l.permute(0, 2, 1)
        l = self.lgn_cv1dlayers(l)
        l = l.squeeze()
        
        print(p.size())
        print(l.size())
        cat = tc.cat((p, l), axis=1).cuda()
        out = self.mlplayers(cat)
        
        return self.regress(out).cuda()
    
#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 100

hp_d['init_learning_rate'] = 10 ** -3.515
hp_d['eps'] = 10 ** -7.178
hp_d['weight_decay'] = 10 ** -4.999

#%% training and validation
kf = KFold(n_splits=10); kf.get_n_splits(shuffled_dataset)
val_err = 0
for tr_idx, ts_idx in kf.split(shuffled_dataset):
    trainset, testset = shuffled_dataset[tr_idx], shuffled_dataset[ts_idx]
    trainset = tuple(trainset); testset = tuple(testset)

    tr_data_loader = DataLoader(trainset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)
    va_data_loader = DataLoader(testset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)

    model = Regressor().to(torch.device('cuda:0'))
    loss_func = nn.MSELoss(reduction='mean').cuda()

    optimizer = optim.Adam(model.parameters(), lr=hp_d['init_learning_rate'], 
        weight_decay=hp_d['weight_decay'], eps=hp_d['eps'])

    tr_epoch_losses = []
    va_epoch_losses = []

    for epoch in range(hp_d['num_epochs']):                             #!! epoch-loop
        # training session
        model.train()
        tr_epoch_loss = 0

        for iter, (seq, smi, label) in enumerate(tr_data_loader):       #!! batch-loop
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
    val_err += min(va_epoch_losses)
print('10-fold CVMSE:', val_err/10)