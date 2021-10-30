import glob
from TurkishStemmer import TurkishStemmer

turk_stem = TurkishStemmer()

print(turk_stem.stem("olta"))
ctr = 0

allTxtFiles = glob.glob(r"C:\Users\erayg\PycharmProjects\NLP-Project\allData/1-*.txt")

for file in allTxtFiles:
    opened = open(file,'r')
    lines = opened.readlines()
    ctr += len(lines)

print(ctr)    #71050 +