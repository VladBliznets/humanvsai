n = 10000
dataset = "dc2"

numbers_of_words = [10, 50, 100, 200, 400, 500, 800, 1000]
coefs_of_ngrams = [0,10,100]
# classifiers = ["Logistic", "LinearSVC", "NuSVC", "BernoulliNB", "MultinomialNB", "SVC"]
classifiers = ["Logistic", "LinearSVC", "NuSVC", "BernoulliNB", "MultinomialNB", "SVC"]

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
from sklearn.svm import LinearSVC, NuSVC, SVC

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
    bgrm = []
    tgrm = []
    fgrm = []
    for ss in result:
        words += ss.copy()
        bgrm += list(nltk.bigrams(ss))
        tgrm += list(nltk.trigrams(ss))
        fgrm += list(nltk.ngrams(ss,4))
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

def extract_features(r):
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
    bgrm = []
    tgrm = []
    fgrm = []
    for ss in result:
        words += ss.copy()
        bgrm += list(nltk.bigrams(ss))
        tgrm += list(nltk.trigrams(ss))
        fgrm += list(nltk.ngrams(ss,4))
    return (words, bgrm, tgrm, fgrm)



train_data_human=[]
train_data_ai=[]
with open('datachat2.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1]=="human":
            train_data_human.append((row[0],row[1]))
        else:
            train_data_ai.append((row[0],row[1]))
train_data_ai=train_data_ai[:n]
train_data_human=train_data_human[:n]
train_data = train_data_ai+train_data_human
random.shuffle(train_data)
print(len(train_data))
words = []
bgrm = []
tgrm = []
fgrm = []
for q in train_data:
    (a,b,c,d) = extract_features(q[0])
    words += a
    bgrm += b
    tgrm += c
    fgrm += d
bgrm_fd = nltk.FreqDist(bgrm)
tgrm_fd = nltk.FreqDist(tgrm)
fgrm_fd = nltk.FreqDist(fgrm)
words = nltk.FreqDist(words)
colloc = []
for number_of_words in tqdm(numbers_of_words):    
    for coef_of_ngrams in coefs_of_ngrams:
        print(coef_of_ngrams)
        for q in (list(bgrm_fd.keys())[:coef_of_ngrams]):
            colloc.append(''.join(q))
        for q in (list(tgrm_fd.keys())[:coef_of_ngrams]):
            colloc.append(''.join(q))
        for q in (list(fgrm_fd.keys())[:coef_of_ngrams]):
            colloc.append(''.join(q))
        word_features = set(list(words.keys())[:number_of_words]+colloc)
        feature_sets=[]
        for r in tqdm(train_data):
            feature_sets.append((features(r[0]),r[1]))
        train_set = feature_sets
        for classifier_used in classifiers:
            Classifier_file_name = str(number_of_words) + "_" + str(coef_of_ngrams) + "_" + classifier_used+"_"+dataset
            if(classifier_used=="Logistic"):
                classifier = SklearnClassifier( LogisticRegression(max_iter=500) )
            if(classifier_used=="SGDC"):
                classifier = SklearnClassifier( SGDClassifier() )
            if(classifier_used=="BernoulliNB"):
                classifier = SklearnClassifier( BernoulliNB() )
            if(classifier_used=="LinearSVC"):
                classifier = SklearnClassifier( LinearSVC() )
            if(classifier_used=="NuSVC"):
                classifier = SklearnClassifier( NuSVC() )
            if(classifier_used=="SVC"):
                classifier = SklearnClassifier( SVC() )
            if(classifier_used=="MultinomialNB"):
                classifier = SklearnClassifier( MultinomialNB() )
            classifier.train(train_set)
            # test_set = feature_sets[16000:]
            cur_path = os.path.dirname(__file__)
            new_path = cur_path+"\\classifiers\\"
            save_classifier = open(new_path+Classifier_file_name,"wb")
            pickle.dump(classifier, save_classifier)
            save_classifier.close()
            save_words = open(new_path+"word_features_"+Classifier_file_name,"wb")
            pickle.dump(word_features, save_words)
            save_words.close()
            # print(Classifier_file_name+": "+str(nltk.classify.accuracy(classifier, test_set)))
            