#stemming stepleri eklenebilir hatta eklenmeli stemmin işe yaramadı gibi gibi
import pandas as pd
from TurkishStemmer import TurkishStemmer
import glob

# change your local adresses with all cleaned data
allTxtFiles = glob.glob(r"C:\Users\erayg\PycharmProjects\NLP-Project\allData/*.txt")

#for file in allTxtFiles:
frames=[]
for file in allTxtFiles:
    dataset = pd.read_csv(file, sep="\t", names=['comment'])

    frames.append(dataset)

dataset = pd.concat(frames, ignore_index=True )

# işlem yapacağım dataset hazır

turkstem = TurkishStemmer()
turkstem.stemWord()
dataset['comment'] = dataset['comment'].str.split()
print(dataset['comment'][0])
dataset['stemmedComment'] = dataset['comment'].apply(lambda x:[turkstem.stem(y) for y in x])
print(turkstem.stem("geldi"))
with open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stemmed.txt",'w') as file:
    for i in range(len(dataset['stemmedComment'])):
        try:
            seperator = " "
            dataset['stemmedComment'][i] = seperator.join(dataset['stemmedComment'][i])
            file.write(dataset['stemmedComment'][i] +"\n")
        except:
            continue