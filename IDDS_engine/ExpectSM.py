import numpy as np
import pandas as pd
import fpkit_modified as fps
from rdkit.Chem import AllChem as chem
import pymysql
pymysql.install_as_MySQLdb()
import _mysql
from sqlalchemy import create_engine
from sm_const import *

#%%
###DB와 연결하는 Class
class CallDB:
    def __init__(self):
        self.engine = create_engine("mysql+mysqldb://{}:{}@{}/{}".format(USER_NAME, USER_PSWD, DB_HOST, DB_NAME_2), encoding='utf-8')
        
    def query_db(self, query, DB_NAME):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=DB_NAME)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        dbconn.commit()
        dbconn.close()
    
    def to_db(self, data, table_nm):
        df = pd.DataFrame(data)
        df.to_sql(name=table_nm, con=self.engine, if_exists='append', index=False)
        
    def from_db(self, query,DB_NAME):
        dbconn = _mysql.connect(host=DB_HOST, user=USER_NAME, passwd=USER_PSWD, port=PORT, db=DB_NAME)
        self.cursor = dbconn.cursor()
        self.cursor.execute(query)
        table = self.cursor.fetchall()
        dbconn.commit()
        dbconn.close()
        return pd.DataFrame(table)

#%%
#DB에서 불러온 데이터 전처리 하는 Class
class PreprocessData:
    def __init__(self, req_id):
        db = CallDB()
        kiba_query =  "SELECT * FROM `kiba_ligand_similarity`;"
        li_query = "SELECT * FROM bd_ligand_info where req_id = '{}'".format(req_id)
        
        self.kiba_fp = db.from_db(kiba_query, DB_NAME_1)
        self.bd_ligand_info = db.from_db(li_query, DB_NAME_2)

    def encoder(self):
        
        kiba_ligand = self.kiba_fp.iloc[:,2].str.split("|")
        kiba_ligand = np.array(kiba_ligand.apply(lambda x:pd.Series(x)), dtype="float64")
                
        return self.kiba_fp, kiba_ligand, self.bd_ligand_info

#%%
#계산 파트
class Calsim:
    def __init__(self):
        self.metrics = ["SM", "RT", "SS2", "CT1", "CT2", "AC", "JT", "Cos", "Dic"]
        
    def calculating(self, ligand, kiba_fp ,kiba_ligand, bd_ligand_info, n=5, similarity="mean",j=1):
        
        m = chem.MolFromSmiles(ligand)
        ecfp_6 = np.array(chem.GetMorganFingerprintAsBitVect(m, 3, nBits=128))
        
        sim_matrix = np.zeros(shape=(len(kiba_ligand) ,10), )
        for i in range(2068):   

            sim={}
            #9개의 metric별 similarity 계산
            for metric in self.metrics:
                sim[metric] = fps.similarity(ecfp_6, kiba_ligand[i,],metric=metric,scale=True)

            sim["mean"] = sum(sim.values())/len(sim.values())
            sim_matrix[i,:] = list(sim.values())

        #sim_matrix 완성 crieteria별로 상위 n개 추출, 이미지 추출
        result = pd.DataFrame(np.concatenate((kiba_fp.iloc[:,0:2], sim_matrix), axis=1))
        result.rename(columns = {0:"chemblID", 1:"SMILES", 2:"SM", 3:"RT", 4:"SS2",
                                 5:"CT1", 6:"CT2", 7:"AC", 8:"JT", 9:"Cos", 10:"Dic",
                                 11:"mean"}, inplace=True)

        final_result = pd.DataFrame(index=range(0,n), columns=['req_id', 'index_id', 'ranking', 'q_chembl_id','t_chembl_id', 'similarity_score'])

        final_result.iloc[:,0] = bd_ligand_info.iloc[0,0]
        final_result.iloc[:,1] = j #index_id
        final_result.iloc[:,2] = list(range(1,6)) #ranking
        #[1]user mole
        final_result.iloc[:,3] = bd_ligand_info.iloc[j-1,2]  #q_chemblid
        final_result.iloc[:,4] = result.sort_values(by=[similarity], ascending=[False]).loc[:,"chemblID"][:n].values #상위 n개 이미지 링크
        #similarity
        final_result.iloc[:,5] = result.sort_values(by=[similarity], ascending=[False]).loc[:,similarity][:n].values #상위 n개 유사도값

        return final_result

#%%
#출력파트
class ExpectSM:
    def __init__(self, req_id):
        self.db = CallDB()  
        self.cal = Calsim()
        pre = PreprocessData(req_id)
        
        self.kiba_fp, self.kiba_ligand, self.bd_ligand_info = pre.encoder()
        self.ligand = self.bd_ligand_info.iloc[:, 3]

    def get_result(self):
        sm_result = pd.DataFrame(columns=['req_id', 'index_id', 'ranking', 'q_chembl_id','t_chembl_id', 'similarity_score'])
        for j in range(len(self.bd_ligand_info)):
            sm_result = sm_result.append(self.cal.calculating(ligand= self.ligand.iloc[j,], kiba_fp = self.kiba_fp ,kiba_ligand = self.kiba_ligand, 
                                                              bd_ligand_info = self.bd_ligand_info, n=5, similarity="mean",j=j+1))
        self.db.to_db(sm_result, "bd_similarity_result")
        file_list = pd.unique(pd.concat([sm_result.iloc[:,3], sm_result.iloc[:,4]]))
        return sm_result, file_list, self.bd_ligand_info










































