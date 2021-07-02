import pickle

with open("trainset_gcn+lstm.txt", "wb") as fp:
    pickle.dump(trainset, fp)
fp.close()
    
with open("trainset_gcn+lstm.txt", "rb") as fp:
    trainset = pickle.load(fp)
fp.close()
    
with open("validset_gcn+lstm.txt", "wb") as fp:
    pickle.dump(validset, fp)
fp.close()
    
with open("validset_gcn+lstm.txt", "rb") as fp:
    validset = pickle.load(fp)
fp.close()
