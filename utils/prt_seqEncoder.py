import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

import torch
from torch import nn

dataset = pd.read_csv('datasets\\kiba.csv')
datalen = len(dataset)

uniprot = list(OrderedDict.fromkeys(list(dataset.loc[:,"uniprotID"])))
sequence = list(OrderedDict.fromkeys(list(dataset.loc[:,"sequence"])))

amino_num = {"A":1, "R":2, "N":3, "D":4, "C":5, "E":6, "Q":7, "G":8, "H":9,
            "I":10, "L":11, "K":12, "M":13, "F":14, "P":15, "S":16, "T":17,
            "W":18, "Y":19, "V":20}

seq_intcode = {}
seq_codelen = {}
for i, seq in enumerate(sequence):
    seq_len = len(seq)
    seq_arr = np.zeros((4128,), dtype=int)
    for j, token in enumerate(seq):
        seq_arr[j] = amino_num[token]
    seq_intcode[uniprot[i]] = seq_arr
    seq_codelen[uniprot[i]] = seq_len

f = open('seq_codeinfo.txt', 'wb')
pickle.dump((seq_intcode, seq_codelen), f)
f.close()

f = open('datasets/dictionaries/prt_lstm.txt', 'rb')
a, b = pickle.load(f)
f.close()

