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
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
from matplotlib import image as img
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import partial

###########################################################################################
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
ligand = np.array(dataset.iloc[:, 9581:-1], dtype=np.int)
kiba = list(dataset['KIBA'])
# del dataset

#%% protein
f = open('datasets/dictionary/seq_codeinfo.txt', 'rb')
seq_intcode, seq_codelen = pickle.load(f)    # dictionary

seq = []
for i in range(datalen):
    seq.append(seq_intcode[protein[i]])
# del protein

seq_len = []
for i in range(datalen):
    seq_len.append(seq_codelen[protein[i]])

#%% ligand ecfp load
ecfp = []
for ec in ligand:
    ecfp.append(ec)
# del ligand

#%% dataset zip
revised_dataset = list(zip(seq, seq_len, ecfp, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

del shuffled_dataset

###########################################################################################
###########################################################################################
#%% mini_dataset loading
with open("datasets/trainset_lstm+ecfp.txt", "rb") as fp:
    trainset = pickle.load(fp)

with open("datasets/validset_lstm+ecfp.txt", "rb") as fp:
    validset = pickle.load(fp)
    
#%% Make collate func.
def collate(samples):
    # The input `samples` is a list of pairs [(graph, label),(graph, label)].
    seqs, seq_lens, ecfps, labels = map(list, zip(*samples))
    return tc.LongTensor(seqs).cuda(), tc.LongTensor(seq_lens), tc.tensor(ecfps, dtype=tc.float).cuda(), tc.tensor(labels).cuda()

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.emlayer = nn.Embedding(21, 10)
        self.lslayer = nn.LSTM(10, 128, num_layers=1, bidirectional=True, batch_first=True)
        
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

    def forward(self, seq, seq_len, ecfp):
        seq_len, sorted_idx = seq_len.sort(0, descending=True)
        seq = seq[sorted_idx]
        
        ls_i = self.emlayer(seq)
        ls_i = pack_padded_sequence(ls_i, seq_len.tolist(), batch_first=True)
        ls_h = torch.zeros(2, 128, 128).cuda()     # (num_layers * num_directions, batch, hidden_size)
        ls_c = torch.zeros(2, 128, 128).cuda()
        
        ls_o, (ls_h, ls_c) = self.lslayer(ls_i, (ls_h, ls_c))
        ls_h = tc.reshape(ls_h.permute(1, 2, 0), (-1, 128))
        
        
        ec_h = self.eclayers(ecfp)
        dim = 1
        for e in ec_h.size()[1:]:
            dim = dim * e
        ec_h = ec_h.view(-1, dim)
        
        cat = tc.cat((ls_h, ec_h), axis=1).cuda()
       
        return self.regress(cat).cuda()

#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 128     
hp_d['num_epochs'] = 300

hp_d['init_learning_rate'] = 10 ** -3.70183
hp_d['eps'] = 10 ** -8.39981
hp_d['weight_decay'] = 10 ** -3.59967

#%% learning and validation
tr_data_loader = DataLoader(trainset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)
va_data_loader = DataLoader(validset, batch_size=hp_d['batch_size'], shuffle=False, collate_fn=collate)

model = Regressor().to(torch.device('cuda:0'))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
loss_func = nn.MSELoss(reduction='mean').cuda()
optimizer = optim.Adam(model.parameters(), lr=hp_d['init_learning_rate'], 
    weight_decay=hp_d['weight_decay'], eps=hp_d['eps'])

print('tr_var:', np.var(np.array([s[3] for s in trainset])))
print('va_var:', np.var(np.array([s[3] for s in validset])))
print('total params:', total_params)

tr_epoch_losses = []
va_epoch_losses = []

start = time.time()

for epoch in range(hp_d['num_epochs']):                          #!! epoch-loop
    # training session
    model.train()
    tr_epoch_loss = 0

    for iter, (seq, seq_len, ecfp, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(seq, seq_len, ecfp).view(-1).cuda()
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

    for iter, (seq, seq_len, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(seq, seq_len, ecfp).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)
