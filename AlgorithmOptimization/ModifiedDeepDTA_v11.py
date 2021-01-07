#%% library import
import numpy as np
import pandas as pd
import networkx as nx
import torch as tc
import torch
import dgl
import pickle
import time

from rdkit.Chem import AllChem as chem
from rdkit.Chem import Descriptors as descriptor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
from dgl import DGLGraph
from dgl import function as fn
from dgl.data.chem import utils
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

###########################################################################################
###########################################################################################
#%% Load dataset and check cuda
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
protein = dataset.loc[:, "uniprotID"]    # 5
ligand = dataset.loc[:, "chemblID"]
kiba = list(dataset.loc[:, 'KIBA'])
del dataset

#%% making protein descriptors
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

#%% making ligand graphs and mol_descriptors
graph = []
for s in ligand:
    graph.append(utils.smiles_to_bigraph(s).to(torch.device('cuda:0')))

#%% dataset zip
revised_dataset = list(zip(desc, graph, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

###########################################################################################
###########################################################################################
#%% mini_dataset loading
with open("datasets/trainset_gcn+desc.txt", "rb") as fp:
    trainset = pickle.load(fp)

with open("datasets/validset_gcn+desc.txt", "rb") as fp:
    validset = pickle.load(fp)

#%% Make collate func.
def collate(samples):
    # The input `samples` is a list of pairs [(graph, label),(graph, label)].
    descs, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs).to(torch.device('cuda:0'))
    return tc.tensor(descs, dtype=tc.float).cuda(), batched_graph, tc.tensor(labels).cuda()
    
#%% GCN module 
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = tc.mean(nodes.mailbox['m'], 1).cuda()
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature.cuda()
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h').cuda()

#%% learning module
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
        
        self.gclayers = nn.ModuleList([
                        GCN(1, 128, F.relu),
                        GCN(128, 256, F.relu),
                        GCN(256, 128, F.relu)]
                        )
               
        self.regress = nn.Linear(256, 1)    # regression

    def forward(self, desc, graph):    # mini-batch, 순서 정확하게 맞음
        ds_h = self.dslayers(desc)
        dim = 1
        for d in ds_h.size()[1:]:
            dim = dim * d
        ds_h = ds_h.view(-1, dim)
        
        gc_h = graph.in_degrees().view(-1, 1).float().cuda()     # 노드의 개수를 feature vector로 사용, ex> torch.Size([7016, 1])
        for conv in self.gclayers:
            gc_h = conv(graph, gc_h).cuda()
        graph.ndata['h'] = gc_h
        
        gc_h = dgl.mean_nodes(graph, 'h')

        cat = tc.cat((gc_h, ds_h),axis=1)
        
        return self.regress(cat)
    
#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 128
hp_d['num_epochs'] = 300

hp_d['init_learning_rate'] = 10 ** -4.4232866
hp_d['eps'] = 10 ** -7.4782489
hp_d['weight_decay'] = 10 ** -4.2857633

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

for epoch in range(hp_d['num_epochs']):                      # epoch-loop

    # training session
    model.train()
    tr_epoch_loss = 0

    for iter, (desc, graph, label) in enumerate(tr_data_loader):  # batch-loop
        prediction = model(desc, graph).view(-1).cuda()
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

    for iter, (desc, graph, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(desc, graph).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
    
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)
