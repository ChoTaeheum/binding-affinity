# The hyperparameters for expection of binding affinity
import os

DB_HOST = "211.233.58.16"
PORT = 3306  #포트번호
USER_NAME = "thcho"
USER_PSWD = "2281"
DB_NAME="ui"

AMINO_SIGN = {"A":1, "R":2, "N":3, "D":4, "C":5, "E":6, "Q":7, "G":8, "H":9,
             "I":10, "L":11, "K":12, "M":13, "F":14, "P":15, "S":16, "T":17,
             "W":18, "Y":19, "V":20}

SMILES_SIGN = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
			  "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
			  "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
			  "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
			  "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
			  "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
			  "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
			  "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# ---------------------paths-----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = os.path.join(BASE_DIR, 'model_weight1.pt')
LOCAL_RES = '/BiO/result/'
REMOTE_RES = '/BiO/Serve/Tomcat/webapps/results/' 

MEAN = 11.719934770879288
STD = 0.8369430908403743