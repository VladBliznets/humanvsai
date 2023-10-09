Classifier_file_name="2.pickle"
number_in_base=0
number_of_words=100
coef_of_ngrams=0
classifier_used="Logistic"
n = 10000

# numbers_of_words = [5000, 10000]
# coefs_of_ngrams = [0, 200]
# "LinearSVC", "NuSVC", "BernoulliNB", "MultinomialNB", "SVC"
# classifiers = ["Logistic"]
classifiers = ["10_0_Logistic_dc2", "10_10_Logistic_dc2", "10_100_Logistic_dc2", "10_0_LinearSVC_dc2", "10_10_LinearSVC_dc2", "10_100_LinearSVC_dc2", "10_0_NuSVC_dc2", "10_10_NuSVC_dc2", "10_100_NuSVC_dc2", "10_0_BernoulliNB_dc2", "10_10_BernoulliNB_dc2", "10_100_BernoulliNB_dc2", "10_0_MultinomialNB_dc2", "10_10_MultinomialNB_dc2", "10_100_MultinomialNB_dc2", "10_0_SVC_dc2", "10_10_SVC_dc2", "10_100_SVC_dc2", "50_0_Logistic_dc2", "50_10_Logistic_dc2", "50_100_Logistic_dc2", "50_0_LinearSVC_dc2", "50_10_LinearSVC_dc2", "50_100_LinearSVC_dc2", "50_0_NuSVC_dc2", "50_10_NuSVC_dc2", "50_100_NuSVC_dc2", "50_0_BernoulliNB_dc2", "50_10_BernoulliNB_dc2", "50_100_BernoulliNB_dc2", "50_0_MultinomialNB_dc2", "50_10_MultinomialNB_dc2", "50_100_MultinomialNB_dc2", "50_0_SVC_dc2", "50_10_SVC_dc2", "50_100_SVC_dc2", "100_0_Logistic_dc2", "100_10_Logistic_dc2", "100_100_Logistic_dc2", "100_0_LinearSVC_dc2", "100_10_LinearSVC_dc2", "100_100_LinearSVC_dc2", "100_0_NuSVC_dc2", "100_10_NuSVC_dc2", "100_100_NuSVC_dc2", "100_0_BernoulliNB_dc2", "100_10_BernoulliNB_dc2", "100_100_BernoulliNB_dc2", "100_0_MultinomialNB_dc2", "100_10_MultinomialNB_dc2", "100_100_MultinomialNB_dc2", "100_0_SVC_dc2", "100_10_SVC_dc2", "100_100_SVC_dc2", "200_0_Logistic_dc2", "200_10_Logistic_dc2", "200_100_Logistic_dc2", "200_0_LinearSVC_dc2", "200_10_LinearSVC_dc2", "200_100_LinearSVC_dc2", "200_0_NuSVC_dc2", "200_10_NuSVC_dc2", "200_100_NuSVC_dc2", "200_0_BernoulliNB_dc2", "200_10_BernoulliNB_dc2", "200_100_BernoulliNB_dc2", "200_0_MultinomialNB_dc2", "200_10_MultinomialNB_dc2", "200_100_MultinomialNB_dc2", "200_0_SVC_dc2", "200_10_SVC_dc2", "200_100_SVC_dc2", "400_0_Logistic_dc2", "400_10_Logistic_dc2", "400_100_Logistic_dc2", "400_0_LinearSVC_dc2", "400_10_LinearSVC_dc2", "400_100_LinearSVC_dc2", "400_0_NuSVC_dc2", "400_10_NuSVC_dc2", "400_100_NuSVC_dc2", "400_0_BernoulliNB_dc2", "400_10_BernoulliNB_dc2", "400_100_BernoulliNB_dc2", "400_0_MultinomialNB_dc2", "400_10_MultinomialNB_dc2", "400_100_MultinomialNB_dc2", "400_0_SVC_dc2", "400_10_SVC_dc2", "400_100_SVC_dc2", "500_0_Logistic_dc2", "500_10_Logistic_dc2", "500_100_Logistic_dc2", "500_0_LinearSVC_dc2", "500_10_LinearSVC_dc2", "500_100_LinearSVC_dc2", "500_0_NuSVC_dc2", "500_10_NuSVC_dc2", "500_100_NuSVC_dc2", "500_0_BernoulliNB_dc2", "500_10_BernoulliNB_dc2", "500_100_BernoulliNB_dc2", "500_0_MultinomialNB_dc2", "500_10_MultinomialNB_dc2", "500_100_MultinomialNB_dc2", "500_0_SVC_dc2", "500_10_SVC_dc2", "500_100_SVC_dc2", "800_0_Logistic_dc2", "800_10_Logistic_dc2", "800_100_Logistic_dc2", "800_0_LinearSVC_dc2", "800_10_LinearSVC_dc2", "800_100_LinearSVC_dc2", "800_0_NuSVC_dc2", "800_10_NuSVC_dc2", "800_100_NuSVC_dc2", "800_0_BernoulliNB_dc2", "800_10_BernoulliNB_dc2", "800_100_BernoulliNB_dc2", "800_0_MultinomialNB_dc2", "800_10_MultinomialNB_dc2", "800_100_MultinomialNB_dc2", "800_0_SVC_dc2", "800_10_SVC_dc2", "800_100_SVC_dc2", "1000_0_Logistic_dc2", "1000_10_Logistic_dc2", "1000_100_Logistic_dc2", "1000_0_LinearSVC_dc2", "1000_10_LinearSVC_dc2", "1000_100_LinearSVC_dc2", "1000_0_NuSVC_dc2", "1000_10_NuSVC_dc2", "1000_100_NuSVC_dc2", "1000_0_BernoulliNB_dc2", "1000_10_BernoulliNB_dc2", "1000_100_BernoulliNB_dc2", "1000_0_MultinomialNB_dc2", "1000_10_MultinomialNB_dc2", "1000_100_MultinomialNB_dc2", "1000_0_SVC_dc2", "1000_10_SVC_dc2", "1000_100_SVC_dc2"] 
# classifiers = ["5000_0_Logistic", "10000_0_Logistic", "5000_200_Logistic", "10000_200_Logistic"]

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
    words=[]
    bgrm=[]
    tgrm=[]
    fgrm=[]
    lemmatizer = WordNetLemmatizer()
    for sentence in sentences_in_quote:  
        words_result = []
        for word in sentence:
            if word not in stop_words:
                words_result.append(lemmatizer.lemmatize(word))
        result.append(words_result)
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



test_data_human=[]
test_data_ai=[]
with open('datachat2.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row[1]=="human":
            test_data_human.append((row[0],row[1]))
        else:
            test_data_ai.append((row[0],row[1]))
test_data_ai=test_data_ai[-n:]
test_data_human=test_data_human[-n:]
test_data = test_data_ai+test_data_human
random.shuffle(test_data)
print(len(test_data))
for classifier_name in classifiers:
    cur_path = os.path.dirname(__file__)
    new_path = cur_path+"\\classifiers\\"
    loadwords = open(new_path+"word_features_"+classifier_name, "rb")
    word_features = pickle.load(loadwords)
    loadwords.close()
    classifier_f = open(new_path+classifier_name, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    feature_sets=[]
    for r in tqdm(test_data):
        feature_sets.append((features(r[0]),r[1]))
    print(classifier_name)
    print(nltk.classify.accuracy(classifier, feature_sets))
