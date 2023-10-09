Classifier_file_name="2.pickle"
number_in_base=0
number_of_words=100
coef_of_ngrams=0
classifier_used="Logistic"
 
import sys
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
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

csv.field_size_limit(sys.maxsize)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def features(r):
    r = ''.join(r.split('\\n'))
    r = ''.join(r.split('\\\''))
    words = []
    sentences_in_quote = []
    for s in sent_tokenize(r.lower()):
        sentences_in_quote.append(word_tokenize(s))
    stop_words = set(stopwords.words("english"))
    result = []
    lemmatizer = WordNetLemmatizer()
    for sentence in sentences_in_quote:  
        words_result = []
        for word in sentence:
            if word not in stop_words:
                words_result.append(lemmatizer.lemmatize(word))
        result.append(words_result)
    words = []
    for ss in result:
        words = ss.copy()
        bgrm = (nltk.bigrams(ss))
        tgrm = (nltk.trigrams(ss))
        fgrm = (nltk.ngrams(ss,4))

    fbigrams=nltk.FreqDist(bgrm)
    ftrigrams=nltk.FreqDist(tgrm)
    ffgrams=nltk.FreqDist(fgrm)   
    feat={}
    for q in list(fbigrams.keys()):
        words.append(''.join(q))
    for q in list(ftrigrams.keys()):
        words.append(''.join(q))
    for q in list(ffgrams.keys()):
        words.append(''.join(q))
    for feature in word_features:
        feat[feature]=(feature in words)
    return feat


test=[]
with open('quora.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        s = ','.join(row)
        if len(s) > 200:
            test.append(s)
print(len(test))
print(test[:5])
random.shuffle(test)
test = test[:10000]

cur_path = os.path.dirname(__file__)
new_path = cur_path+"\\classifiers\\"
loadwords = open(new_path+"word_features_"+Classifier_file_name, "rb")
word_features = pickle.load(loadwords)
loadwords.close()
classifier_f = open(new_path+Classifier_file_name, "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

feature_sets=[]
for r in tqdm(test):
    feature_sets.append((features(r),"human"))

print(nltk.classify.accuracy(classifier, feature_sets))
