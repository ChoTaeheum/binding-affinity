from rdkit.Chem import AllChem as chem
from rdkit.Chem import Draw as draw
import numpy as np
import pandas as pd
from collections import OrderedDict

dataset = pd.read_csv('datasets/KIBA.csv')
ligands = dataset.loc[:, ['chemblID', 'SMILES']]
ligands.duplicated()
ligands = ligands.drop_duplicates(['chemblID'])
mols = {chID:chem.MolFromSmiles(smiles) for idx, (chID, smiles) in ligands.iterrows()}

for i, (c, m) in enumerate(mols.items()):
    draw.MolToFile(m,'datasets/dictionaries/ligand_img/{}.png'.format(c))       
        