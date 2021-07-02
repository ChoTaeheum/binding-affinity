# The hyperparameters for expection of binding affinity
import os

DB_HOST = "192.168.0.50"
PORT = 3306  #포트번호
USER_NAME = "jjh"
USER_PSWD = "93"
DB_NAME_1 = "binding_db"
DB_NAME_2 = "ui"
TABLE_1 = "kiba_ligand_similarity"
TABLE_2 = "bd_ligand_info"
TABLE_3 = "bd_similarity_result2"

#DB_LIGAND_LIST = os.listdir("/var/lib/tomcat9/webapps/kiba_ligand")

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
