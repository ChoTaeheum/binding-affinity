import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
import torch
from torch import nn

dataset = pd.read_csv('datasets/KIBA.csv')
ligands = dataset.loc[:, ['chemblID', 'SMILES']]
ligands.duplicated()
ligands = ligands.drop_duplicates(['chemblID'])

smiles_num = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

smi_intcode = {}
for idx, (che, smi) in ligands.iterrows():
    smi_arr = np.zeros((590,), dtype=int)
    for j, token in enumerate(smi):
        smi_arr[j] = smiles_num[token]
    smi_intcode[che] = smi_arr

f = open('datasets/dictionaries/lgn_smiecoding.txt', 'wb')
pickle.dump(smi_intcode, f)
f.close()

f = open('datasets/dictionaries/lgn_smiecoding.txt', 'rb')
a, b = pickle.load(f)
f.close()
