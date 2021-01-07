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

sequence = sequence[:, :2000]

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
    return tc.FloatTensor(sequences).cuda(), tc.FloatTensor(smileseq, type=tc.float).cuda(), tc.tensor(labels).cuda()

#%% network module 선언
class BasicConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
    
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        
        self.stem = nn.Sequential(
            BasicConv1d(1, 16, kernel_size=3, stride=2),
            BasicConv1d(16, 16, kernel_size=3, stride=1),
            BasicConv1d(16, 32, kernel_size=3, stride=1, padding=1)
            )
        
        self.leaf0 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.leaf1 = BasicConv1d(32, 48, kernel_size=3, stride=2)
        
        self.branch0 = nn.Sequential(
            BasicConv1d(80, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 48, kernel_size=3, stride=1)
            )
        
        self.branch1 = nn.Sequential(
            BasicConv1d(80, 32, kernel_size=1, stride=1),
            BasicConv1d(32, 32, kernel_size=7, stride=1),
            BasicConv1d(32, 48, kernel_size=3, stride=1, padding=3)
            )
        
        self.leaf2 = BasicConv1d(96, 96, kernel_size=3, stride=2)
        
        self.leaf3 = nn.MaxPool1d(kernel_size=3, stride=2)
        
    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.leaf0(x0)
        x2 = self.leaf1(x0)
        x3 = torch.cat((x1, x2), axis=1)
        x4 = self.branch0(x3)
        x5 = self.branch1(x3)
        x6 = torch.cat((x4, x5), axis=1)
        x7 = self.leaf2(x6)
        x8 = self.leaf3(x6)
        out = torch.cat((x7, x8), axis=1)
        return out
        
        
class InceptionResnet_Ablock(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResnet_Ablock, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv1d(192, 16, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(192, 16, kernel_size=1, stride=1),
            BasicConv1d(16, 16, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(192, 16, kernel_size=1, stride=1),
            BasicConv1d(16, 24, kernel_size=3, stride=1, padding=1),
            BasicConv1d(24, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv1d = nn.Conv1d(64, 192, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv1d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
    

class Reduction_Ablock(nn.Module):
    def __init__(self):
        super(Reduction_Ablock, self).__init__()

        self.branch0 = BasicConv1d(192, 192, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv1d(192, 128, kernel_size=1, stride=1),
            BasicConv1d(128, 128, kernel_size=3, stride=1, padding=1),
            BasicConv1d(128, 192, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionResnet_Bblock(nn.Module):
    def __init__(self, scale=1.0):
        super(InceptionResnet_Bblock, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv1d(576, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(576, 64, kernel_size=1, stride=1),
            BasicConv1d(64, 80, kernel_size=1, stride=1),
            BasicConv1d(80, 96, kernel_size=7, stride=1, padding=3)
        )

        self.conv1d = nn.Conv1d(192, 576, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv1d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out
    

class Reduction_Bblock(nn.Module):
    def __init__(self):
        super(Reduction_Bblock, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv1d(576, 128, kernel_size=1, stride=1),
            BasicConv1d(128, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv1d(576, 128, kernel_size=1, stride=1),
            BasicConv1d(128, 144, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv1d(576, 128, kernel_size=1, stride=1),
            BasicConv1d(128, 144, kernel_size=3, stride=1, padding=1),
            BasicConv1d(144, 160, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool1d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    
class InceptionResnet_Cblock(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super(InceptionResnet_Cblock, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv1d(1072, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(1072, 96, kernel_size=1, stride=1),
            BasicConv1d(96, 112, kernel_size=1, stride=1),
            BasicConv1d(112, 128, kernel_size=3, stride=1, padding=1)
        )

        self.conv1d = nn.Conv1d(224, 1072, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv1d(out)
        out = out * self.scale + x
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channel):
        super(SqueezeExcitation, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        out = self.squeeze(x).view(b, c)
        out = self.excitation(out).view(b, c, 1)
        out = x * out.expand_as(x)
        return out
    
#%% learning module 선언
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.

        self.inc_res_se_layers = nn.Sequential(
                        Stem(),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=192),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=192),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=192),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=192),
                        InceptionResnet_Ablock(scale=0.17),
                        SqueezeExcitation(channel=192),
                        Reduction_Ablock(), 
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        InceptionResnet_Bblock(scale=0.10),
                        SqueezeExcitation(channel=576),
                        Reduction_Bblock(),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=1072),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=1072),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=1072),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=1072),
                        InceptionResnet_Cblock(scale=0.20),
                        SqueezeExcitation(channel=1072)
                        )
        
        self.p_avgpool = nn.AvgPool1d(61, count_include_pad=False)

        self.l_avgpool = nn.AvgPool1d(4, count_include_pad=False)
        
        self.mlplayers = nn.Sequential(
                        nn.Linear(2144, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.8)
                        )

        self.regress = nn.Linear(1024, 1)    # regression

    def forward(self, prt_seq, lgn_seq):
        p = prt_seq.unsqueeze(1)
        p = self.inc_res_se_layers(p)
        p = self.p_avgpool(p).squeeze()
        
        l = lgn_seq.unsqueeze(1)
        l = self.inc_res_se_layers(l)
        l = self.l_avgpool(l).squeeze()
        
        cat = tc.cat((p, l), axis=1).cuda()
        out = self.mlplayers(cat)
        
        return self.regress(out).cuda()

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
np.save('ModifiedDeepDTA_v3_tr_losses', tr_epoch_losses)
np.save('ModifiedDeepDTA_v3_va_losses', va_epoch_losses)
min(va_epoch_losses)