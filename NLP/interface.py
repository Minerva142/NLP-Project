import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from TurkishStemmer import TurkishStemmer
from PIL import Image
from string import digits
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle




def pre_process(text):
    text = text.lower()
    text = re.sub("", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    return text

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

def Button_Clicked_Calculation_wot_stemming(text,keyword_num,max_df,vocab_count):
    turkstem = TurkishStemmer()
    seperator = " "
    allTxtFiles = glob.glob(r"C:\Users\erayg\PycharmProjects\NLP-Project\allData/*.txt")

    # for file in allTxtFiles:
    frames = []
    for file in allTxtFiles:
        dataset = pd.read_csv(file, sep="\t", names=['stemmedComment'])
        dataset = dataset[dataset.stemmedComment != "this order was cancelled as requested by the user as it exceeded the average delivery time"]
        frames.append(dataset)

    dataset = pd.concat(frames, ignore_index=True)
    docs = dataset['stemmedComment'].tolist()

    cv = CountVectorizer(max_df=max_df,max_features=vocab_count)
    word_count_vector=cv.fit_transform(docs)
    # tf-ıdf steps

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names=cv.get_feature_names()

    stopWords = []
    fileWords = open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stopwords",'r',encoding='windows-1254')
    stopWords = fileWords.readlines()
    for i in range(len(stopWords)):
        stopWords[i] = stopWords[i].split("\n")[0]

    #burada verilen input cümlesi temizlenmeli, gerekli temizlemelerden sonra kulanılabilir
    doc = text.lower()
    doc = re.sub(r'((http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?)','',doc)
    doc = re.sub(r'[^\w\s]', '', doc)
    remove_digits = str.maketrans('', '', digits)
    doc = doc.translate(remove_digits)
    doc = doc.split()
    doc = list(filter(None, doc))
    doc = seperator.join(doc)

    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(feature_names,sorted_items,keyword_num*2)
    realKeywords = {}
    ctr = 0
    for k in keywords:
        if k not in stopWords:
            if len(k) > 2:
                ctr += 1
                realKeywords.update({k : keywords[k]})
        if ctr == keyword_num:
            break
    return realKeywords

def Button_Clicked_Calculation_with_stemming(text,keyword_num,max_df,vocab_count):
    turkstem = TurkishStemmer()
    seperator = " "
    file = r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stemmed.txt"
    dataset = pd.read_csv(file, sep="\t", names=['stemmedComment'], engine='python', encoding='windows-1254')

    dataset['stemmedComment'] = dataset['stemmedComment'].apply(lambda x: pre_process(x))
    docs = dataset['stemmedComment'].tolist()

    cv = CountVectorizer(max_df=max_df,max_features=vocab_count)
    word_count_vector=cv.fit_transform(docs)

    # tf-ıdf steps

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    feature_names=cv.get_feature_names()

    stopWords = []
    fileWords = open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stopwords",'r',encoding='windows-1254')
    stopWords = fileWords.readlines()
    for i in range(len(stopWords)):
        stopWords[i] = stopWords[i].split("\n")[0]

    #burada verilen input cümlesi temizlenmeli, gerekli temizlemelerden sonra kulanılabilir
    doc = text.lower()
    doc = re.sub(r'((http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?)','',doc)
    doc = re.sub(r'[^\w\s]', '', doc)
    remove_digits = str.maketrans('', '', digits)
    doc = doc.translate(remove_digits)
    doc = doc.split()
    doc = list(filter(None, doc))
    docWords = []
    for word in doc:
        if word not in stopWords:
            docWords.append(turkstem.stem(word))

    doc = seperator.join(docWords)

    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(feature_names,sorted_items,keyword_num*2)
    realKeywords = {}
    ctr = 0
    for k in keywords:
        if k not in stopWords:
            if len(k) > 2:
                ctr += 1
                realKeywords.update({k : keywords[k]})
        if ctr == keyword_num:
            break
    return realKeywords


def preprocess_sentence(text):
    seperator = " "
    doc = text.lower()
    doc = re.sub(r'((http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?)', '',
                 doc)
    doc = re.sub(r'[^\w\s]', '', doc)
    remove_digits = str.maketrans('', '', digits)
    doc = doc.translate(remove_digits)
    doc = doc.split()
    doc = list(filter(None, doc))
    fileWords = open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stopwords", 'r', encoding='windows-1254')
    stopWords = fileWords.readlines()
    for i in range(len(stopWords)):
        stopWords[i] = stopWords[i].split("\n")[0]
    docWords = []
    for word in doc:
        if word not in stopWords:
            docWords.append(word)

    doc = seperator.join(docWords)
    return doc

def button_clicked_NER(text):
    seperator = " "
    model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
    tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
    ner = pipeline('ner', model=model, tokenizer=tokenizer)
    doc = text.lower()
    doc = re.sub(r'((http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?)', '',
                 doc)
    doc = re.sub(r'[^\w\s]', '', doc)
    remove_digits = str.maketrans('', '', digits)
    doc = doc.translate(remove_digits)
    doc = doc.split()
    doc = list(filter(None, doc))
    stopWords = []
    fileWords = open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\stopwords", 'r', encoding='windows-1254')
    stopWords = fileWords.readlines()
    for i in range(len(stopWords)):
        stopWords[i] = stopWords[i].split("\n")[0]
    docWords = []
    for word in doc:
        if word not in stopWords:
            docWords.append(word)

    doc = seperator.join(docWords)
    result = ner(doc)

    s = ""
    l = list()
    for i in result:
        if "B" in i["entity"]:
            l.append(s)
            s = i["word"]
        elif "I" in i["entity"]:
            if "##" in i["word"]:
                s = s + i["word"].strip("##")
            else:
                s = s + " " + i["word"]
        # print(s)
    l.append(s)

    return l

# Model de buraya gelecek




st.title("TAB Gıda için NLP tabanlı Proje")

st.header("Verilerimiz nereden elde ettik")
st.write("""

1. [Yemek Sepeti](https://www.yemeksepeti.com)

2. [Şikayet Var](https://www.sikayetvar.com/)

3. [Ekşi Sözlük](https://eksisozluk.com/)

4. [Twitter](https://twitter.com/)

""")
st.header("Yorumlara Genel Bakış")
st.write("Toplam yorum sayısı: __123754__")


# Second
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

# DUMB DATA
image = Image.open(r"C:\Users\erayg\PycharmProjects\NLP-Project\images\pie_chart_veri.PNG")
st.image(image, caption = 'Yorumların Dağılımı', width = 1000)

# harita kısmı
st.title("Yemek Sepeti Dashboard")

col1, col2 = st.columns(2)
image_yemek = Image.open(r"C:\Users\erayg\PycharmProjects\NLP-Project\images\yemeksepeti-logo.jpeg")
col2.image(image_yemek,width = 200)

restaurantType=col2.selectbox("Restaurant Adı",["Arbys","Burger King","Popeyes","Sbarro"])

if restaurantType=="Sbarro":
    fileName=r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\sbarro-YemekSepeti.csv"
    df = pd.read_csv(fileName)

    restaurantName=col2.selectbox("Şube Adı",list(df["fileName"]))

    st.header("Seçilen Restaurant için Yorumlar ve Puanlar")
    commment_df=pd.read_json("data/Sbarro-Yemek_Sepeti/"+restaurantName)
    st.dataframe(commment_df)

    map_df=df[df["fileName"]==restaurantName].filter(['lat','lon'], axis=1)
    col1.map(map_df)

st.header("Yorumları Test Etme")
sentence = st.text_input('Lütfen denemek istediğiniz cümleyi giriniz')
# diğer model de buraya gömülecek
loaded_model = pickle.load(open(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\NlpModel_new.pkl", 'rb'))

# df_labelled okunacak
df_labelled = pd.read_csv(r"C:\Users\erayg\PycharmProjects\NLP-Project\NLP\labeled_data.csv")

def predict_sentiment2(sentence,loaded_model,df_labelled):
    docs_new = []
    docs_new.append(sentence)
    count_vect = CountVectorizer()
    sentence_list = df_labelled['yorum'].to_list()
    X_train_counts = count_vect.fit_transform(sentence_list)
    X_new_counts = count_vect.transform(docs_new)
    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    # sentiment_names = ['nagatif', 'nötr', 'pozitif']
    sentiment_names = ['negatif', 'pozitif']

    predicted = loaded_model.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, sentiment_names[category]))
        return sentiment_names[category]


# ön ayar olarak yukarda modele vocab count, keyword count ve max_df bilgisi alınacak
max_df = st.slider("max_df değerini seçin",0.0,3.0,step=0.01)
keyword_count = st.slider("Anahtar kelime sayısını girin",1,5)
vocab_count = st.number_input("Sözlüğünüzde kaç kelime olsun istiyorsunuz:",min_value=0,max_value=50000)
# button eklenilecek
col1, col2, col3, col4 = st.columns(4)
if st.button("Analiz Et"):
    # burada diğer modelden gelen labelı da göstericez onu fonksiyonunu da çağırıp
    with col1:
        st.header("Duygu Analizi")
        # diğer model labeli buraya gelecek
        # sentence ufak bi temizlenecek(punctuation at, sayı çıkart yeterli) kodu yukarda var zaten
        cleaned_sentence = preprocess_sentence(sentence)
        sentiment_names = predict_sentiment2(cleaned_sentence,loaded_model,df_labelled)
        st.write(sentiment_names)
    keywords = Button_Clicked_Calculation_wot_stemming(sentence,keyword_count,max_df,vocab_count)
    with col2:
        st.header("Anahtar Kelimeler(ayıklanmamış)")
        for key in keywords:
            st.write(key + "  :   " + str(keywords[key]))
    keywords_stem = Button_Clicked_Calculation_with_stemming(sentence, keyword_count, max_df, vocab_count)
    with col3:
        st.header("Anahtar Kelimeler(ayıklanmış)")
        for key in keywords_stem:
            st.write(key + "  :   " + str(keywords_stem[key]))
    NER = button_clicked_NER(sentence)
    with col4:
        st.header("İsim varlık tanımları")
        for element in NER:
            st.write(element)

# sonuçların görüneceği bir panel olacak


# görsel eklenilecek(word cloud)
image = Image.open(r"C:\Users\erayg\PycharmProjects\NLP-Project\images\bk_logo_wc.png")
st.image(image, caption = 'Yorumlardan oluşturulmuş kelime bulutu')