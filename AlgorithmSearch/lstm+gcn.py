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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import partial

###########################################################################################
###########################################################################################
#%% Load dataset and check cuda
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
protein = dataset.loc[:, 'uniprotID']
ligand = dataset.loc[:, 'chemblID']
kiba = list(dataset['KIBA'])
del dataset

#%% protein sequence load
f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
seq_voc, seq_len = pickle.load(f)
f.close()

sequence = np.zeros(((2**13)+(2**8), 4128))
for i, s in enumerate(protein):
    sequence[i] = seq_voc[s]
    
sequence_len = np.zeros(((2**13)+(2**8),))
for i, s in enumerate(protein):
    sequence_len[i] = seq_len[s]
    
#%% ligand graph load
f = open('datasets/dictionaries/lgn_smiles.txt', 'rb')
smiles = pickle.load(f)
f.close()

graph = []
for i, c in enumerate(ligand):
    graph.append(utils.smiles_to_bigraph(smiles[c]).to(torch.device('cuda:0')))

#%% dataset zip
revised_dataset = list(zip(sequence, sequence_len, graph, kiba))
shuffled_dataset = shuffle(revised_dataset); del revised_dataset
trainset = shuffled_dataset[:2**13]
validset = shuffled_dataset[2**13:(2**13) + (2**8)]

del shuffled_dataset

#%% Make collate func.
def collate(samples):
    sequences, sequence_lens, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs).to(torch.device('cuda:0'))
    return tc.LongTensor(sequences).cuda(), tc.LongTensor(sequence_lens), batched_graph, tc.tensor(labels).cuda()
    
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
        
        self.emlayer = nn.Embedding(21, 10)
        self.lslayer = nn.LSTM(10, 64, num_layers=1, bidirectional=True, batch_first=True)
        
        self.gclayers = nn.ModuleList([
                        GCN(1, 128, F.relu),
                        GCN(128, 256, F.relu),
                        GCN(256, 128, F.relu)]
                        )

        self.regress = nn.Linear(256, 1)    # regression

    def forward(self, seq, seq_len, graph):    # mini-batch, 순서 정확하게 맞음
        sorted_seq_len, sorted_idx = seq_len.sort(0, descending=True)
        seq = seq[sorted_idx]
        
        ls_i = self.emlayer(seq)
        ls_i = pack_padded_sequence(ls_i, sorted_seq_len.tolist(), batch_first=True)
        ls_h = torch.zeros(2, 128, 64).cuda()     # (num_layers * num_directions, batch, hidden_size)
        ls_c = torch.zeros(2, 128, 64).cuda()
        
        ls_o, (ls_h, ls_c) = self.lslayer(ls_i, (ls_h, ls_c))
        ls_o, _ = pad_packed_sequence(ls_o, batch_first=True)

        # 순서 다시 바로잡아주기        
        _, sortedback_idx = sorted_idx.sort(0)
        ls_o = ls_o[sortedback_idx]
        
        # 각 sample의 last output vector 추출
        for_o = []
        for idx, o in enumerate(ls_o):
            for_o.append(o[seq_len[idx]-1, :64].view(1, 64))
        for_o = torch.cat(for_o, 0)
        back_o = ls_o[:, 0, 64:]
        concat_o = tc.cat((for_o, back_o), axis=1)   # batch, hidden*2
        
        gc_h = graph.in_degrees().view(-1, 1).float().cuda()     # 노드의 개수를 feature vector로 사용, ex> torch.Size([7016, 1])
        for conv in self.gclayers:
            gc_h = conv(graph, gc_h).cuda()
        graph.ndata['h'] = gc_h
        
        gc_h = dgl.mean_nodes(graph, 'h')

        cat = tc.cat((concat_o, gc_h),axis=1)
        
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

print('tr_var:', np.var(np.array([s[3] for s in trainset])))
print('va_var:', np.var(np.array([s[3] for s in validset])))
print('total params:', total_params)

tr_epoch_losses = []
va_epoch_losses = []

start = time.time()

for epoch in range(hp_d['num_epochs']):                      # epoch-loop
    # training session
    model.train()
    tr_epoch_loss = 0

    for iter, (seq, seq_len, graph, label) in enumerate(tr_data_loader):  # batch-loop
        prediction = model(seq, seq_len, graph).view(-1).cuda()
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

    for iter, (seq, seq_len, graph, label) in enumerate(va_data_loader):  # batch-loop
        prediction = model(seq, seq_len, graph).view(-1).cuda()
        loss = loss_func(prediction, label).cuda()
        
        va_epoch_loss += loss.detach().item()
    
    va_epoch_loss /= (iter + 1)
    print('Validation epoch {}, loss {:.4f}'.format(epoch, va_epoch_loss))
    va_epoch_losses.append(va_epoch_loss)
    
end = time.time()
print('time elapsed:', end-start)
