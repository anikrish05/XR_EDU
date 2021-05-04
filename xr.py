from flask import Flask, render_template
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from numpy import average
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit')
def submit():
	new_input =[['']]
  #new_output = clf.predict(new_input)
	tfidf_testInput=tfidf_vectorizer.transform(new_input[0])
	new_output = clf.predict(tfidf_testInput)
	new_output


