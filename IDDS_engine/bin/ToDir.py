import os
import pandas as pd
import xlsxwriter
import pymysql
pymysql.install_as_MySQLdb()
import _mysql
import shutil

from ba_const import *
from sm_const import *
from ExpectBA import CallDB
from rdkit.Chem import AllChem as chem
from rdkit.Chem import Draw as draw
from sqlalchemy import create_engine
from datetime import datetime



class ToDir:
    def __init__(self, req_id, ba_result, sm_result, file_list, ligand_info):
        #######!!! making directory
        
        os.system('mkdir '+ SAVERES + '/' + req_id)                  # req_dir
        os.system('mkdir '+ SAVERES + '/' + req_id + '/' + 'png')    # png_dir
        
        #######!!! summary result
        self.call_db = CallDB()
        
        summary_query = """
        SELECT ri.req_id,
               (SELECT cd_val1
                  FROM ui.com_cdkey1
        	     WHERE dvsn_cd = 'R1000'
                   AND cd_key1 = ri.req_type) AS req_type,
               (SELECT cd_val1
                  FROM ui.com_cdkey1
        	     WHERE dvsn_cd = 'R2000'
        		   AND cd_key1 = ri.req_sub_type) AS req_type,
               bui.protein_num,
               bli.ligand_num,
               ri.req_des
          FROM ui.req_info AS ri,
               (SELECT req_id, COUNT(uniprot_index) AS protein_num
                  FROM ui.bd_uniprot_info) AS bui,
               (SELECT req_id, COUNT(ligand_index) AS ligand_num
                  FROM ui.bd_ligand_info) AS bli,
               (SELECT req_id, stat_cd, error_cd
                  FROM ui.req_status_info
                 WHERE hstry_stat_cd = 'O') AS rsi
         WHERE ri.req_id = bui.req_id
           AND bui.req_id = bli.req_id
           AND ri.req_id = rsi.req_id
           AND ri.req_id = 'RQ00000001'
           AND ri.hstry_stat_cd = 'O'
           """
           
        prt_query = """
        select ui.uniprot_id, pd.protein_des, ui.protein_seq
        from ui.bd_uniprot_info as ui
        left join protein_db.uniprot_info as pd
        on ui.uniprot_id = pd.uniprot_id
        where ui.req_id = (SELECT ui.req_id FROM ui.req_info order by ui.req_id desc limit 1)
        """

        lgn_query = """
        select ui.chembl_id, bd.chembl_nm, ui.smiles
        from ui.bd_ligand_info as ui
        left join binding_db.kiba_ligand_db as bd
        on ui.chembl_id = bd.chembl_id
        where ui.req_id = (SELECT ui.req_id FROM ui.req_info order by ui.req_id desc limit 1)
        """
        
        self.insert = """
        insert into ui.req_status_info
        values ('{}',
		(SELECT max(exec_seq)+1 FROM ui.req_status_info ALIAS_FOR_SUBQUERY where req_id='{}'),
        (select cd_key1 from ui.com_cdkey1 where cd_val1='Complete'),
        null,
        (select start_dt from ui.req_status_info ALIAS_FOR_SUBQUERY where req_id='{}'),
        now(),
        'O')
        """.format(req_id, req_id, req_id)
        
        self.update = """
        update ui.req_status_info
        set hstry_stat_cd = 'H'
        where req_id='{}' and stat_cd=110
        """.format(req_id)
        
        self.summary = self.call_db.from_db(summary_query).T
        self.protein = self.call_db.from_db(prt_query)
        self.ligand = self.call_db.from_db(lgn_query)
        self.req_id = req_id
        
        self.protein.columns = ['Uniprot_id', 'Protein_name', 'AminoAcid_seq']
        self.ligand.columns = ['Chembl_id', 'ligand_name', 'SMILES']
        self.summary = self.summary.rename(columns={0:'info'}, 
                                      index={0:'req_id', 1:'req_type', 2:'req_subtype', 
                                             3:'no_protein', 4:'no_ligand', 5:'req_description'})
        
        #######!!! smilarity info, image
        self.ligand_info = ligand_info
        self.kiba_ligand_db = pd.DataFrame(index = range(1), columns = ["chembl_id", "chembl_nm", "inchikey", "smiles", "ecfp"])
        self.db_ligand_list = os.listdir("/var/lib/tomcat9/webapps/kiba_ligand")
        self.file_list = file_list
        
        self.dbsave()
        self.imagetofolder()
        
        #######!!! ba_result, sm_result -> conclude
        self.ba_result = ba_result
        self.sm_result = sm_result
        
        self.to_excel()
        self.to_zip()
        self.update_stat()
        
    def dbsave(self):
        for i in range(len(self.ligand_info)):
            if self.ligand_info.iloc[i,2]+".png" not in self.db_ligand_list:
                kiba_ligand_db_tmp = pd.DataFrame(index = range(1), columns = ["chembl_id", "chembl_nm", "inchikey", "smiles", "ecfp"])
                kiba_ligand_db_tmp.iloc[0,0] = self.ligand_info.iloc[i,2]   #new ligand ID
                kiba_ligand_db_tmp.iloc[0,3] = self.ligand_info.iloc[i,3]   #new ligand smiles
                self.kiba_ligand_db  = self.kiba_ligand_db.append(kiba_ligand_db_tmp)
                
               #이미지 저장    
                m = chem.MolFromSmiles(self.ligand_info.iloc[i,3] )
                draw.MolToFile(m, "/var/lib/tomcat9/webapps/kiba_ligand/" + self.ligand_info.iloc[i,2] + ".png".format()) 
        
        self.kiba_ligand_db = self.kiba_ligand_db.iloc[1:,:]
        engine = create_engine("mysql+mysqldb://jjh:"+"93"+"@192.168.0.50/binding_db", encoding='utf-8')
        self.kiba_ligand_db.to_sql(name="kiba_ligand_db", con=engine, if_exists='append', index=False)
        
    def imagetofolder(self):
        from_directory = "/var/lib/tomcat9/webapps/kiba_ligand"
        destination_directory = SAVERES + self.req_id + '/png'
        
        for i in range(len(self.file_list)):
            shutil.copy(from_directory + "/" + self.file_list[i] + ".png", destination_directory)
        
    def to_excel(self):
        today = datetime.today()
        date = today.strftime('%Y%m%d')
        
        writer = pd.ExcelWriter(SAVERES + self.req_id + '/' + self.req_id + '_BA_' + date + '.xlsx', engine = 'xlsxwriter')
        workbook = writer.book
        
        worksheet = workbook.add_worksheet('summary')
        writer.sheets['summary'] = worksheet
        self.summary.to_excel(writer, sheet_name='summary', startrow=1, startcol=1)
        self.protein.to_excel(writer, sheet_name='summary', startrow=9, startcol=1)
        self.ligand.to_excel(writer, sheet_name='summary', startrow=11+len(self.protein), startcol=1)
        
        worksheet = workbook.add_worksheet('BA_results')
        writer.sheets['BA_results'] = worksheet
        self.ba_result.to_excel(writer, sheet_name='BA_result', startrow=1, startcol=1)
        
        worksheet = workbook.add_worksheet('SM_results')
        writer.sheets['SM_results'] = worksheet
        self.sm_result.to_excel(writer, sheet_name='SM_result', startrow=1, startcol=1)
        
        writer.save()
        
    def to_zip(self):
        os.chdir(SAVERES)
        os.system('zip -r ' + self.req_id + '.zip ./' + self.req_id + '/*')
        
    def update_stat(self):
        self.call_db.query_db(self.insert)
        self.call_db.query_db(self.update)
