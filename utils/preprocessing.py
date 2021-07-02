#%% library import
import numpy as np
import pandas as pd
import networkx as nx
import torch as tc
import torch
import pprint
import pickle

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

#%% Load dataset and cuda
dataset = pd.read_csv("D:\\a_projects\\datasets\\KIBA.csv")
datalen = len(dataset)
cuda = tc.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
#%%
protein = np.array(dataset.iloc[:, 5:1582], dtype=float)    # 5
ligand = "images\\kiba\\"
kiba = list(dataset['KIBA'])

#%% protein descriptor normalization, 현재 information leaking 나중에 수정 요망
prt_ds = {}
prt_ds[0] = protein[:, 0:20] # AAC
prt_ds[1] = protein[:, 20:420] #DC
prt_ds[2] = protein[:, 420:660] #AC1
prt_ds[3] = protein[:, 660:900] #AC2
prt_ds[4] = protein[:, 900:1140] #AC3
prt_ds[5] = protein[:, 1140:1161] #C
prt_ds[6] = protein[:, 1161:1182] #T
prt_ds[7] = protein[:, 1182:1287] #D
prt_ds[8] = protein[:, 1287:1317] #SOCN
prt_ds[9] = protein[:, 1317:1347] #SOCN
prt_ds[10] = protein[:, 1347:1367] #QSO
prt_ds[11] = protein[:, 1367:1387] #QSO
prt_ds[12] = protein[:, 1387:1417] #QSO
prt_ds[13] = protein[:, 1417:1447] #QSO
prt_ds[14] = protein[:, 1447:1467] #PAAC
prt_ds[15] = protein[:, 1467:1497] #PAAC
prt_ds[16] = protein[:, 1497:1517] #APAAC
prt_ds[17] = protein[:, 1517:1577] #APAAC

for i, ds in enumerate(prt_ds.values()):
    mean = np.mean(ds)
    std = np.std(ds)
    prt_ds[i] = (ds - mean) / std
    
nor_protein = np.array((), dtype=np.float64).reshape(118254,0)
for i, ds in enumerate(prt_ds.values()):
    nor_protein = np.hstack([nor_protein, ds])
    
desc = []
for ds in nor_protein:
    desc.append(ds)
    
#%% ligand ecfp, image load
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
image = np.zeros((2**13, 1, 280, 280))
for i in range(len(image)):
    im = img.imread(ligand + "kiba_{}.png".format(i))
    image[i][0] = rgb2gray(im)[10:290, 10:290]
    
#%%
revised_dataset = list(zip(desc, image, kiba))
shuffled_dataset = shuffle(revised_dataset)
trainset = shuffled_dataset[:2**11]
validset = shuffled_dataset[2**11:(2**11) + (2**10)]

#del prt_ds, nor_protein, revised_dataset, dataset
###################################################################################################################################
###################################################################################################################################
