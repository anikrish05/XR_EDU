from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from numpy import average
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
app = Flask(__name__)


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
	class_labels = classifier.classes_
	feature_names = vectorizer.get_feature_names()
	topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
	topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
	data = []

	for coef, feat in topn_class1:
		print(class_labels[0], coef, feat)
		data.append([feat, coef])

		print()

	for coef, feat in reversed(topn_class2):
		print(class_labels[1], coef, feat)
		data.append([feat, coef])

		print(topn_class1)
		df = pd.DataFrame(data, columns = ['Word', 'Value'])
		print(topn_class1)
		print(df)
		sns.set(rc={'figure.figsize':(20,8.27)})
		graph = sns.barplot(x = 'Word', y = 'Value', data=df, ci=65)
		graph

df_fake=pd.read_csv('Fake.csv')
df_true=pd.read_csv('True.csv')


	# add a 'label' column filled with the booleans False and True for the fake news and true news datasets respectively
df_fake['label'] = False
df_true['label'] = True
	# combines and randomizes the two datasets
combine_set = pd.concat([df_fake,df_true]).sample(frac =  1,random_state = 1)


	# Gets the labels from the label column
labels = combine_set.label

	# equally splits the combined dataset into test and training datasets; 
	# separates text and label columns from the combined data sets
	# uses 80% of each divided part of the dataset as a training set 
	# the other 20%s become test sets 
	# randomizes divided parts of dataset 7 times
x_train,x_test,y_train,y_test=train_test_split(combine_set['text'], labels, test_size=0.2, random_state=7)

	# this basically starts a feature extraction command that turns text documents into TF-IDF features
	# TF-IDF features are bascially words that have been given a numerical value to represent how significant they are
	# stop_words='english' removes all english stop words from the text i.e. "a", "an", "the", "be", etc.
	# max_df = 0.7 makes it so that all words that appear in more than 70% of the given documents are not included when making the features
	# it filters these because at a certain point of commonality, certain words just become noise for the model like stop words; they don't help distinguish False articles from True articles 
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
	# Basically applies the above filters to the test and training sets for one category of label
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

	# Initialize the `count_vectorizer` 
count_vectorizer = CountVectorizer(stop_words='english')

	# Fit and transform the training data.
count_train = count_vectorizer.fit_transform(x_train)

	# Transform the test set 
count_test = count_vectorizer.transform(x_test)
	# Get the feature names of `tfidf_vectorizer` 
print(tfidf_vectorizer.get_feature_names()[-10:])
	# Get the feature names of `count_vectorizer` 
print(count_vectorizer.get_feature_names()[0:10])
	# initializes a Passive Agressive Classifier machine learning algorithm and assigns it to "pac"; makes it so that this model will go over training data 50 times; 
	# the more times a model runs on a training set, the more accurate it is, but it is important not to overfit the model, so 50 is a reasonable number of times to run it through the training set
	# passive-aggressive algorithms are a family of machine learning algorithms for large-scale learning(perfect for our datasets with 20,000+ articles)
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
# predict on the test set and calculate accuracy, recall, precision, and f1
y_pred=pac.predict(tfidf_test)
print(f'Accuracy: {round(accuracy_score(y_test,y_pred)*100,2)}%')
print(f'Recall: {round(recall_score(y_test,y_pred)*100,2)}%')
print(f'Precision: {round(precision_score(y_test,y_pred)*100,2)}%')
print(f'F1: {round(f1_score(y_test,y_pred)*100,2)}%')
clf = MultinomialNB() 
clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = accuracy_score(y_test, pred)
print(f'Accuracy: {round(accuracy_score(y_test,pred)*100,2)}%')
print(f'Recall: {round(recall_score(y_test,pred)*100,2)}%')
print(f'Precision: {round(precision_score(y_test,pred)*100,2)}%')
print(f'F1: {round(f1_score(y_test,pred)*100,2)}%')
clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
score = accuracy_score(y_test, pred)
print(f'Accuracy: {round(accuracy_score(y_test,pred)*100,2)}%')
print(f'Recall: {round(recall_score(y_test,pred)*100,2)}%')
print(f'Precision: {round(precision_score(y_test,pred)*100,2)}%')
print(f'F1: {round(f1_score(y_test,pred)*100,2)}%')
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
score = accuracy_score(y_test, pred)
print(f'Accuracy: {round(accuracy_score(y_test,pred)*100,2)}%')
print(f'Recall: {round(recall_score(y_test,pred)*100,2)}%')
print(f'Precision: {round(precision_score(y_test,pred)*100,2)}%')
print(f'F1: {round(f1_score(y_test,pred)*100,2)}%')
	#This function will go inside submit, before that though pull tfidf_vectorizer and linear_clf 
	#from the csv files and load them up
most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=10)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit', methods=['GET', "POST"])
def submit():
	new_input =[[request.form['userText']]]
  #new_output = clf.predict(new_input)
	tfidf_testInput=tfidf_vectorizer.transform(new_input[0])
	new_output = clf.predict(tfidf_testInput)
	new_data=new_output.tolist()

	return render_template('index.html', value=(json.dumps(new_data)[1:len(json.dumps(new_data))-1]))


