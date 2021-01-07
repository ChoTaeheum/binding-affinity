from rdkit.Chem import AllChem as chem
from rdkit.Chem import Draw as draw
import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle

dataset = pd.read_csv('datasets/KIBA.csv')
ligands = dataset.loc[:, ['chemblID', 'SMILES']]
ligands.duplicated()
ligands = ligands.drop_duplicates(['chemblID'])
smiles = {chID:smiles for idx, (chID, smiles) in ligands.iterrows()}

f = open('datasets/dictionaries/lgn_smiles.txt', 'wb')
pickle.dump((smiles), f)
f.close()

f = open('datasets/dictionaries/lgn_ecfp.txt', 'rb')
smiles = pickle.load(f)
f.close()