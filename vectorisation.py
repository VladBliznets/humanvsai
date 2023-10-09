train_param="dc2"
test_param="dc2"

import csv
import nltk
import os
import re
import random
import pickle
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import chain
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import sklearn.feature_extraction
from sklearn import *
from sklearn.svm import LinearSVC, NuSVC, SVC

n=70000
m=5000
if(train_param=="dc1"):
    n=1500
if(test_param=="dc1"):
    m=1500
if(train_param=="peg"):
    n=7000
if(test_param=="peg"):
    m=7000

if(train_param==test_param and train_param=="dc1"):
    n=120
    m=1000
if(train_param==test_param and train_param=="peg"):
    n=5000
    m=2000
    # sklearn.feature_extraction.text.HashingVectorizer(),sklearn.feature_extraction.text.TfidfVectorizer(), sklearn.feature_extraction.text.CountVectorizer(),
vectorizers=[sklearn.feature_extraction.text.HashingVectorizer(ngram_range=(1,5)),sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,5)), sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,5)),sklearn.feature_extraction.text.HashingVectorizer(),sklearn.feature_extraction.text.TfidfVectorizer(), sklearn.feature_extraction.text.CountVectorizer() ]
classifiers=[ sklearn.linear_model.LogisticRegression(max_iter=1000), sklearn.linear_model.PassiveAggressiveClassifier() , sklearn.linear_model.SGDClassifier(), sklearn.linear_model.RidgeClassifier()]
#sklearn.linear_model.LogisticRegression(max_iter=500), sklearn.linear_model.PassiveAggressiveClassifier() , sklearn.linear_model.SGDClassifier(), sklearn.linear_model.RidgeClassifier(),sklearn.ensemble.AdaBoostClassifier(),sklearn.tree.DecisionTreeClassifier(), sklearn.ensemble.GradientBoostingClassifier(),sklearn.ensemble.RandomForestClassifier(),     sklearn.ensemble.GradientBoostingClassifier(), sklearn.tree.DecisionTreeClassifier(), sklearn.ensemble.AdaBoostClassifier(),
data_human=[]
data_ai=[]
csv_name=""
if(train_param=="dc1"):
    csv_name="datachat1.csv"
if(train_param=="dc2"):
    csv_name="datachat2.csv"
if(train_param=="peg"):
    csv_name="Pegasus.csv"
with open(csv_name, 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1]=="human":
            data_human.append((row[0],row[1]))
        else:
            data_ai.append((row[0],row[1]))
train_data_ai=data_ai[:n]
train_data_human=data_human[:n]
train_data = train_data_ai+train_data_human
random.shuffle(train_data)

data_human=[]
data_ai=[]
if(test_param=="dc1"):
    csv_name="datachat1.csv"
if(test_param=="dc2"):
    csv_name="datachat2.csv"
if(test_param=="peg"):
    csv_name="Pegasus.csv"
with open(csv_name, 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1]=="human":
            data_human.append((row[0],row[1]))
        else:
            data_ai.append((row[0],row[1]))
test_data_ai=data_ai[-m:]
test_data_human=data_human[-m:]
test_data = test_data_ai+test_data_human
texts=[]
labels=[]
for t in train_data:
    texts.append(t[0])
    labels.append(t[1])
texts_test=[]
labels_test=[]
for t in test_data:
    texts_test.append(t[0])
    labels_test.append(t[1])
for vectorizer in vectorizers:
    vectors=vectorizer.fit_transform(texts)
    vectors_test=vectorizer.transform(texts_test)
    for classifier in classifiers:
        classifier.fit(vectors,labels)
        print("!"+str(vectorizer)+" "+str(classifier)+" "+str(classifier.score(vectors_test,labels_test))+"!\n")
        
