import os
import pandas as pd
import numpy as np
from collections import OrderedDict
dataset = pd.read_csv('datasets\\kiba.csv')
datalen = len(dataset)

#%%###########################################################################
#one hot encoding값 뽑아내는 함수1
def one_hot_encoding1(sequence):
       #index 부여
       aminonum = {"A":0, "R":1, "N":2, "D":3, "C":4, "E":5, "Q":6, "G":7, "H":8,
                   "I":9, "L":10, "K":11, "M":12, "F":13, "P":14, "S":15, "T":16,
                   "W":17, "Y":18, "V":19}
       
       one_hot_vector = [0]*(20)
       index = aminonum[sequence]
       one_hot_vector[index] = 1
       return one_hot_vector

#%%###########################################################################
#"one hot encoding값 뽑아내는 함수1"을 sequence 길이만큼 반복해서 출력하는 함수
def one_hot_encoding2(sequence):
    token = list(sequence) 

    #빈 행렬 생성
    final = np.zeros((20, len(token)), dtype=np.int64)
    
    #빈 공간 채우기
    for i in range(len(token)):
        final[:,i] = one_hot_encoding1(token[i])
    
    return final

#"one hot encoding2" 함수를 각 sequence마다 적용해서 행렬 형태로 뽑아내기.
    


#%%###########################################################################
def one_hot_encoding3(data):
    #protein 고유한 값만 추출
    uniprot = list(OrderedDict.fromkeys(list(data.loc[:,"uniprotID"])))
    data = list(OrderedDict.fromkeys(list(data.loc[:,"sequence"])))
    
    #각 protein별로 작업 진행(output: dict 형태)
    prs = {}
    for i, pr in enumerate(data):
        prs[uniprot[i]] = one_hot_encoding2(pr)
    
    return prs
   
#%%############################################################################
#함수 입력 끝, 함수 사용
#함수 사용

encoding = one_hot_encoding3(dataset)
#%%############################################################################ 
#저장하기   
import pickle
pickle.dump(prs,file)

with open('encoding.pickle', 'wb') as f:
    pickle.dump(encoding, f, pickle.HIGHEST_PROTOCOL)
    
    
    
#%%
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

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

f = open('seq_codeinfo.txt', 'rb')
a, b = pickle.load(f)
f.close()

