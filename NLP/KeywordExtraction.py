import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from TurkishStemmer import TurkishStemmer



def pre_process(text):
    text = text.lower()
    text = re.sub("", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    return text

turkstem = TurkishStemmer()
seperator = " "
file = r"C:\Users\erayg\OneDrive\Masaüstü\ödtü dönemler\4.sınıf 2.dönem\ceng499\hw4\hw4_files\stemmed.txt"
dataset = pd.read_csv(file, sep="\t", names=['stemmedComment'],engine='python',encoding='windows-1254')

dataset['stemmedComment'] = dataset['stemmedComment'].apply(lambda x:pre_process(x))
docs = dataset['stemmedComment'].tolist()


cv = CountVectorizer(max_df=0.5,max_features=20000)
word_count_vector=cv.fit_transform(docs)

# tf-ıdf steps

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

feature_names=cv.get_feature_names()

doc = "baya kötü bir yemekti, patatesler buz gibiydi"
doc = doc.split()
docWords = []
for word in doc:
    docWords.append(turkstem.stem(word))

doc = seperator.join(docWords)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

stopWords = []
fileWords = open(r"C:\Users\erayg\OneDrive\Masaüstü\ödtü dönemler\4.sınıf 2.dönem\ceng499\hw4\hw4_files\stopwords",'r',encoding="utf-8")
stopWords = fileWords.readlines()
for i in range(len(stopWords)):
    stopWords[i] = stopWords[i].split("\n")[0]

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
sorted_items=sort_coo(tf_idf_vector.tocoo())
keywords=extract_topn_from_vector(feature_names,sorted_items,5)
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
realKeywords = []
for k in keywords:
    if k not in stopWords:
        if len(k) > 2:
            realKeywords.append(k)
            print(k,keywords[k])
