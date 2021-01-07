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
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from matplotlib import image as img
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

###########################################################################################
###########################################################################################
#%% Load dataset and cuda
dataset = pd.read_csv("datasets/KIBA.csv")
cuda = tc.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
#%% protein-ligand-kiba split
protein = np.array(dataset.iloc[:, 4:9581], dtype=float)    # 5
ligand = np.array(dataset.iloc[:, 9581:-1], dtype=np.int)
kiba = list(dataset['KIBA'])

#%% protein descriptor normalization, 현재 information leaking 나중에 수정 요망
prt_ds = {}
prt_ds[0] = protein[:, 0:20]       # AAC
prt_ds[1] = protein[:, 20:420]     # DC
prt_ds[2] = protein[:, 420:8420]   # TC
prt_ds[3] = protein[:, 8420:8660]  # AC1
prt_ds[4] = protein[:, 8660:8900]  # AC2
prt_ds[5] = protein[:, 8900:9140]  # AC3
prt_ds[6] = protein[:, 9140:9161]  # C
prt_ds[7] = protein[:, 9161:9182]  # T
prt_ds[8] = protein[:, 9182:9287]  # D
prt_ds[9] = protein[:, 9287:9317]  # SOCN
prt_ds[10] = protein[:, 9317:9347] # SOCN
prt_ds[11] = protein[:, 9347:9367] # QSO
prt_ds[12] = protein[:, 9367:9387] # QSO
prt_ds[13] = protein[:, 9387:9417] # QSO
prt_ds[14] = protein[:, 9417:9447] # QSO
prt_ds[15] = protein[:, 9447:9467] # PAAC
prt_ds[16] = protein[:, 9467:9497] # PAAC
prt_ds[17] = protein[:, 9497:9517] # APAAC
prt_ds[18] = protein[:, 9517:9577] # APAAC
del protein

for i, ds in enumerate(prt_ds.values()):
    mean = np.mean(ds)
    std = np.std(ds)
    prt_ds[i] = (ds - mean) / std
    
nor_protein = np.array((), dtype=np.float64).reshape(118254,0)
for i, ds in enumerate(prt_ds.values()):
    nor_protein = np.hstack([nor_protein, ds])
del prt_ds
    
desc = []
for ds in nor_protein:
    desc.append(ds)
del nor_protein
    
#%% ligand ecfp load
ecfp = []
for ec in ligand:
    ecfp.append(ec)

#%% dataset zip
revised_dataset = list(zip(desc, ecfp, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

del shuffled_dataset

###########################################################################################
###########################################################################################
#%% mini_dataset loading
with open("datasets/trainset_ecfp+desc.txt", "rb") as fp:
    trainset = pickle.load(fp)

with open("datasets/validset_ecfp+desc.txt", "rb") as fp:
    validset = pickle.load(fp)
    
#%% Make collate func.
def collate(samples):
    # The input `samples` is a list of pairs [(graph, label),(graph, label)].
    descs, ecfps, labels = map(list, zip(*samples))
    return tc.tensor(descs, dtype=tc.float).cuda(), tc.tensor(ecfps, dtype=tc.float).cuda(), tc.tensor(labels).cuda()

#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.dslayers = nn.Sequential(     
                        nn.Linear(9577, 2048),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(2048, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 128),
                        )
        
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

    def forward(self, desc, ecfp):
        ds_h = self.dslayers(desc)
        dim = 1
        for d in ds_h.size()[1:]:
            dim = dim * d
        ds_h = ds_h.view(-1, dim)
            
        ec_h = self.eclayers(ecfp)
        dim = 1
        for e in ec_h.size()[1:]:
            dim = dim * d
        ec_h = ec_h.view(-1, dim)
        
        cat = tc.cat((ds_h, ec_h), axis=1).cuda()
       
        return self.regress(cat).cuda()
    
#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 32     # 512
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

    for iter, (desc, ecfp, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(desc, ecfp).view(-1).cuda()
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

    for iter, (desc, ecfp, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(desc, ecfp).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)
