#%% library import
import numpy as np
import pandas as pd
import networkx as nx
import torch as tc
import time
import subprocess

import pymysql
pymysql.install_as_MySQLdb()
import _mysql

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sqlalchemy import create_engine
from ba_const import *

#%%
class CallDB:
    def __init__(self):
        self.engine = create_engine("mysql+mysqldb://{}:{}@{}/{}".format(USER_NAME, USER_PSWD, DB_HOST, DB_NAME), encoding='utf-8')
    
    def query_db(self, query):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=DB_NAME)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        dbconn.commit()
        dbconn.close()
    
    def to_db(self, data, table_nm):
        df = pd.DataFrame(data)
        df.to_sql(name=table_nm, con=self.engine, if_exists='append', index=False)
        
    def from_db(self, query):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=DB_NAME)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        table = self.cursor.fetchall()
        dbconn.commit()
        dbconn.close()
        return pd.DataFrame(table)
    
#%%
class PreprocessData:
    def __init__(self, req_id):
        call_db = CallDB()
        pr_query = "SELECT req_id, uniprot_id, protein_seq FROM ui.bd_uniprot_info where req_id = '{}'".format(req_id)
        li_query = "SELECT req_id, chembl_id, smiles FROM ui.bd_ligand_info where req_id = '{}'".format(req_id)
        kiba_query = "select * from kiba_score"
        
        self.prt_df = call_db.from_db(pr_query).iloc[:, 1:]
        self.lgn_df = call_db.from_db(li_query).iloc[:, 1:]

    def encoder(self):
        prtlen = len(self.prt_df)
        lgnlen = len(self.lgn_df)
        
        prt_seqs = np.zeros((prtlen, 1400), dtype=int)
        lgn_seqs = np.zeros((lgnlen, 100), dtype=int)
        
        for i, row in enumerate(self.prt_df.iterrows()):
            prt_seq = row[1][2][:1400]
            for j, t in enumerate(prt_seq):
                prt_seqs[i, j] = AMINO_SIGN[t]
        del i, j, t, prt_seq
        
        for i, row in enumerate(self.lgn_df.iterrows()):
            lgn_seq = row[1][2][:100]
            for j, t in enumerate(lgn_seq):
                lgn_seqs[i, j] = SMILES_SIGN[t]
        del i, j, t, lgn_seq

        prt_ids = np.array(self.prt_df.iloc[:, 0])
        lgn_ids = np.array(self.lgn_df.iloc[:, 0])
        
        prt_data = np.reshape(np.array([], dtype=int), (0, 1400))
        lgn_data = np.reshape(np.array([], dtype=int), (0, 100))
        for prt_seq in prt_seqs:
            for lgn_seq in lgn_seqs:
                prt_data = np.vstack((prt_data, prt_seq))
                lgn_data = np.vstack((lgn_data, lgn_seq))
                
        prt_label = []
        lgn_label = []
        for prt_id in prt_ids:
            for lgn_id in lgn_ids:
                prt_label.append(prt_id)
                lgn_label.append(lgn_id)
        id_index = pd.DataFrame([prt_label, lgn_label]).T
                
        return prt_data, lgn_data, id_index

#%%
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()    # method 상속받고 __init__()은 여기서 하겠다.
        
        self.prt_emlayer = nn.Embedding(21, 10)
        
        self.prt_cv1dlayers = nn.Sequential(
                        nn.Conv1d(10, 32, kernel_size = 4),
                        nn.BatchNorm1d(num_features = 32),
                        nn.ReLU(),
                        nn.Conv1d(32, 64, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.Conv1d(64, 96, kernel_size = 12),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=1379)
                        )
        
        ######################################################################
        ######################################################################
        
        self.lgn_emlayer = nn.Embedding(64, 10)
        
        self.lgn_cv1dlayers = nn.Sequential(
                        nn.Conv1d(10, 32, kernel_size = 4),
                        nn.BatchNorm1d(num_features = 32),
                        nn.ReLU(),
                        nn.Conv1d(32, 64, kernel_size = 6),
                        nn.BatchNorm1d(num_features = 64),
                        nn.ReLU(),
                        nn.Conv1d(64, 96, kernel_size = 8),
                        nn.BatchNorm1d(num_features = 96),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size = 85)
                        )

        self.mlplayers = nn.Sequential(
                        nn.Linear(192, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU()
                        )

        self.regress = nn.Linear(512, 1)    # regression

    def forward(self, prt_seq, lgn_seq):   
        p = self.prt_emlayer(prt_seq)
        p = p.permute(0, 2, 1)
        p = self.prt_cv1dlayers(p)
        p = p.squeeze()
        p = p.view(-1, p.size()[-1])
        print(p.size())
        
        l = self.lgn_emlayer(lgn_seq)
        l = l.permute(0, 2, 1)
        l = self.lgn_cv1dlayers(l)
        l = l.squeeze()
        l = l.view(-1, l.size()[-1])
        print(l.size())
        
        cat = tc.cat((p, l), axis=1).cuda()
        print(cat.size())
        
        out = self.mlplayers(cat)
        
        return self.regress(out).cuda()

#%%
class ExpectBA:
    def __init__(self, req_id):
        self.req_id = req_id
        preprocess = PreprocessData(self.req_id)
        prt_data, lgn_data, self.id_index = preprocess.encoder()
        smplen = len(self.id_index)
                
        model = Regressor().to(tc.device('cuda:0'))
        optimizer = optim.Adam(model.parameters(), lr=(10**-3.563), weight_decay=(10**-5), eps=(10**-8))
        
        checkpoint = tc.load(CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        model.eval()
        
        prt_data = tc.tensor(prt_data, dtype=tc.int64).cuda()
        lgn_data = tc.tensor(lgn_data, dtype=tc.int64).cuda()

        prediction = np.array(model(prt_data, lgn_data).detach().cpu())
        self.prediction = pd.DataFrame((prediction-MEAN) / STD)
        
        self.table = pd.DataFrame({'req_id': self.req_id,
                                   'index_id': np.arange(1, smplen+1),
                                   'uniprot_id': self.id_index.iloc[:, 0],
                                   'chembl_id': self.id_index.iloc[:, 1],
                                   'ba_score': self.prediction.iloc[:, 0]
                                   })
        
        self.to_db()
    
    def to_db(self):
        db = CallDB()
        db.to_db(self.table, 'bd_result')
        
    def get_result(self):
        return self.req_id, self.table


    




