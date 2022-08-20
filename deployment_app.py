#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# loading the trained model
pickle_in = open(r"D:\DA_and_DS_class\Data_science\DS_project\rf_model.pkl", 'rb') 
model = pickle.load(pickle_in)

pickle_in = open(r"D:\DA_and_DS_class\Data_science\DS_project\vectorizer.pkl", 'rb') 
vectorizer = pickle.load(pickle_in)

# Title of the application
st.header("Predict Ratings for Hotel Reviews")
st.subheader("Enter the review to analyze")

input_text = st.text_area("Type review here", height=50)

option = st.sidebar.selectbox('Menu bar',['Sentiment Analysis','Keywords'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
       
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open(r"D:\DA_and_DS_class\Data_science\DS_project\rf_model.pkl", 'rb') 
        model = pickle.load(pickle_in)
        pickle_in = open(r"D:\DA_and_DS_class\Data_science\DS_project\vectorizer.pkl", 'rb') 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        
        if model.predict(transformed_input) == -1:
            st.write(" Negative 😔")
        elif    model.predict(transformed_input) == 1:
            st.write("Positive 😃")
            # st.balloons()
        else:
            st.write(" Neutral 😶")
        

elif option == "Keywords":
    st.header("Keywords")
    if st.button("Keywords"):
        
        r=Rake(language='english') #RAKE: Rapid Automatic Keyword Extraction
        r.extract_keywords_from_text(input_text)
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Display the important phrases
        st.write("These are the **keywords** causing the above sentiment:")
        for i, p in enumerate(phrases):
            st.write(i+1, p)


# In[2]:

st.snow()
# In[3]:
# st.subheader(' Created By : PROJECT GROUP NO ')

# In[ ]: