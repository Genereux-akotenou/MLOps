import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from tensorflow.keras.models import load_model
import pickle

stop_words = stopwords.words("english")
stemmer = SnowballStemmer('english')

def nettoyage(text):
    to_remove = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(to_remove, " ", str(text).lower().strip())
    tokens = []
    
    for token in text.split():
        if token not in stop_words:
            token = stemmer.stem(token)
            tokens.append(token)
            
    return " ".join(tokens)

@st.cache_resource
def chargement_model():
    global model
    model = load_model('model.h5')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        global tokenizer
        tokenizer = pickle.load(handle)
    return tokenizer

model = chargement_model()
tokenizer = load_tokenizer()

st.title('Analyse de Sentiment')

tweet = st.text_area("Entrez votre tweet...")

if tweet:
    tweet = nettoyage(tweet)
    st.write(f"Votre tweet nettoyé donne ceci : {tweet}")
    tweet = tokenizer.texts_to_sequences([tweet])
    tweet = pad_sequences(tweet, maxlen=30)
    result = model.predict([tweet])
    proba = result.item() * 100
    st.write(f"Votre tweet est posifif avec une proba de : {proba:.2f} %")
    if proba > 50:
        st.write('Votre tweet est positif')
    else:
        st.write('Votre tweet est négatif')
