import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from TurkishStemmer import TurkishStemmer
from PIL import Image
from string import digits
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import glob




def pre_process(text):
  text = text.lower()
  text = re.sub("", "", text)
  text = re.sub("(\\d|\\W)+", " ", text)
  return text


allTxtFiles = glob.glob(r"C:\Users\erayg\PycharmProjects\NLP-Project\allData/*.txt")

#for file in allTxtFiles:
frames=[]
for file in allTxtFiles:
    dataset = pd.read_csv(file, sep="\t", names=['stemmedComment'])
    # dataseti elden geçir, ingilizce cümleyi çıkar (önemli işşşş)
    dataset = dataset[dataset.stemmedComment != "this order was cancelled as requested by the user as it exceeded the average delivery time"]
    frames.append(dataset)

dataset = pd.concat(frames, ignore_index=True )
docs = dataset['stemmedComment'].tolist()


cv = CountVectorizer(max_df=0.85,max_features=20000)
word_count_vector=cv.fit_transform(docs)


def get_top_n_words(corpus, n=None):
  vec = CountVectorizer().fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0)
  words_freq = [(word, sum_words[0, idx]) for word, idx in
                vec.vocabulary_.items()]
  words_freq = sorted(words_freq, key=lambda x: x[1],
                      reverse=True)
  return words_freq[:n]


# Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(docs, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns = ["Word", "Freq"]
# Barplot of most freq words
sns.set(rc={'figure.figsize': (13, 8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]
top2_words = get_top_n2_words(docs, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)
plt.show()

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
feature_names=cv.get_feature_names()

doc = "getirden burger king namıkkemal şubesine sipariş verdim gelen ürünler alakasız büyük boy patates bile küçücük geldi umursama yok lakayıt bir şubeymiş söylediklerim doğru değil getir i aradım oralı değiller ilgilenmediler bile geçiştirdi"

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

from scipy.sparse import coo_matrix


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


