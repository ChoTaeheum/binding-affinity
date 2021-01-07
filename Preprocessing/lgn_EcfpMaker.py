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
mols = {chID:chem.MolFromSmiles(smiles) for idx, (chID, smiles) in ligands.iterrows()}

ecfp = {}
for i, (c, m) in enumerate(mols.items()):
    ecfp[c] = np.array(chem.GetMorganFingerprintAsBitVect(m, 3, nBits=512))

f = open('datasets/dictionaries/lgn_ecfp(512).txt', 'wb')
pickle.dump((ecfp), f)
f.close()

f = open('datasets/dictionaries/lgn_ecfp(512).txt', 'rb')
ecfp = pickle.load(f)
f.close()