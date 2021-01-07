#%% library import
import numpy as np
import pandas as pd
import networkx as nx
import torch as tc
import torch
import dgl
import pickle
import time
import mdtraj

from rdkit.Chem import AllChem as chem
from rdkit.Chem import Descriptors as descriptor
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from dgl import DGLGraph
from dgl import function as fn
from dgl.data.chem import utils
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

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

#%% ligand ecfp and graph load
f = open('datasets/dictionaries/lgn_ecfp.txt', 'rb')
ecfp = pickle.load(f)
f.close()

ecfprint = np.zeros(((2**16)+(2**13), 2048))
for i, c in enumerate(ligand):
    ecfprint[i] = ecfp[c]

f = open('datasets/dictionaries/lgn_smiles.txt', 'rb')
smiles = pickle.load(f)
f.close()

graph = []
for i, c in enumerate(ligand):
    graph.append(utils.smiles_to_bigraph(smiles[c]).to(torch.device('cuda:0')))
    
#%% dataset zip
revised_dataset = list(zip(sequence, ecfprint, graph, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**16]
validset = shuffled_dataset[2**16:(2**16) + (2**13)]

del shuffled_dataset

#%% Make collate func.
def collate(samples):
    sequences, ecfprints, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs).to(torch.device('cuda:0'))
    return tc.LongTensor(sequences).cuda(), tc.tensor(ecfprints, dtype=tc.float).cuda(), batched_graph, tc.tensor(labels).cuda()

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
    
#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.emlayer = nn.Embedding(21, 10)
        
        self.cv2dlayer = nn.Sequential(
                        nn.Conv2d(1, 8, kernel_size = (64, 10), stride=(2, 1)),
                        nn.BatchNorm2d(num_features = 8),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = (2, 1)),
                        nn.Dropout(0.2)
                        )
        
        self.cv1dlayers = nn.Sequential(
                        nn.Conv1d(8, 16, kernel_size = 3),
                        nn.BatchNorm1d(num_features = 16),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 2),
                        nn.Dropout(0.2),
                        nn.Conv1d(16, 24, kernel_size = 2),
                        nn.BatchNorm1d(num_features = 24),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 2),
                        nn.Dropout(0.2)
                        )
            
        self.lslayer = nn.LSTM(24, 64, num_layers=1, bidirectional=True, batch_first=True)
        
        self.eclayers = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 64),
                        )
        
        self.gclayers = nn.ModuleList([
                        GCN(1, 64, F.relu),
                        GCN(64, 128, F.relu),
                        GCN(128, 64, F.relu)]
                        )

        self.regress = nn.Linear(256, 1, F.elu)    # regression

    def forward(self, seq, ecfp, graph):        
        cv_i = self.emlayer(seq)
        
        cv2_i = cv_i.unsqueeze(1)
        cv2_o = self.cv2dlayer(cv2_i)
        cv1_i = cv2_o.squeeze()
        cv1_o = self.cv1dlayers(cv1_i)
        
        ls_i = cv1_o.permute(0, 2, 1)
        
        ls_h = torch.zeros(2, 64, 64).cuda()     # (num_layers * num_directions, batch, hidden_size)
        ls_c = torch.zeros(2, 64, 64).cuda()
        
        ls_o, (ls_h, ls_c) = self.lslayer(ls_i, (ls_h, ls_c))

        for_o = ls_o[:, -1, :64]
        back_o = ls_o[:, 0, 64:]
        
        concat_o = tc.cat((for_o, back_o), axis=1)   # batch, hidden*2
        
        ec_h = self.eclayers(ecfp)
        dim = 1
        for e in ec_h.size()[1:]:
            dim = dim * e
        ec_h = ec_h.view(-1, dim)
        
        gc_h = graph.in_degrees().view(-1, 1).float().cuda()     # 노드의 개수를 feature vector로 사용, ex> torch.Size([7016, 1])
        for conv in self.gclayers:
            gc_h = conv(graph, gc_h).cuda()
        graph.ndata['h'] = gc_h
        gc_h = dgl.mean_nodes(graph, 'h')
        
        cat = tc.cat((concat_o, ec_h, gc_h), axis=1).cuda()
       
        return self.regress(cat).cuda()

#%% Set hyperparameter
hp_d = {}

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 64
hp_d['num_epochs'] = 200

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

    for iter, (seq, ecfp, graph, label) in enumerate(tr_data_loader):  #!! batch-loop
        prediction = model(seq, ecfp, graph).view(-1).cuda()
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

    for iter, (seq, ecfp, graph, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(seq, ecfp, graph).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
        
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)

#%% 저장
np.save('results/cnnlstm+ecfpgcn_tr_losses_v3', tr_epoch_losses)
np.save('results/cnnlstm+ecfpgcn_va_losses_v3', va_epoch_losses)