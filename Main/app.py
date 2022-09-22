
from flask import Flask, render_template, url_for, request, jsonify,Response
from flask.helpers import flash
from nltk.corpus.reader.reviews import TITLE      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import text
import textblob
import pickle   
from textblob import TextBlob
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import plotly
from plotly import offline
import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from plotly.offline import iplot
import cufflinks
from sklearn.metrics.pairwise import cosine_similarity 
import re
import wordcloud
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import string



app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"X.pkl", "rb") as f:
    X = pickle.load(f)

with open(r"V.pkl", "rb") as f:
    V = pickle.load(f)

with open(r"FILE.pkl", "rb") as f:
    FILE = pickle.load(f)

with open(r"TITLE.pkl", "rb") as f:
    title = pickle.load(f)
with open(r"Orig.pkl", "rb") as f:
    Orig = pickle.load(f)

index_list=[]
similarity=[]
def calculate_similarity(X, vectorizor, query, top_k=5):

    
    
    query_vec = vectorizor.transform(query)
    
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)
def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    
    counter = 1
    for index in similar_doc_indices:
        #pred_title.append("Title: "+str(title[index])+", Similarity: "+str(round(cosine_similarities[index],2)*100))
        index_list.append(index)
        similarity.append(str(round(cosine_similarities[index],2)*100))
        print('Top-{}, Similarity = {}'.format(counter, cosine_similarities[index]))
        print('Document: {}, .txt'.format(index + 1))
        counter += 1


# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    
        # Take a string input from user'
    index_list.clear()
    similarity.clear()

    user_input = request.form['text']
    data=[user_input]
    print(data)
    sim_vecs, cosine_similarities = calculate_similarity(X, V, data,top_k=20)
    show_similar_documents(Orig, cosine_similarities, sim_vecs)
    sel_titles=[]
    sel_doc=[]
    for i in index_list:
        sel_titles.append(title[i])
        sel_doc.append(FILE[i])
    print(sel_titles)
    D=[]
    for index_ in range(len(sel_titles)):
        D.append((sel_titles[index_],similarity[index_],index_list[index_]))

    return render_template('index.html', titles =D )

@app.route("/download/<label>", methods=['GET'])
def show_document(label=None):
    return render_template('document.html',doc=Orig[int(label)],Title=title[int(label)])

@app.route("/download_file/<label>", methods=['GET'])
def download_doc(label=None):
    csv = str(Orig[int(label)])
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename="+label+".txt"})


     
app.run(debug=True)

