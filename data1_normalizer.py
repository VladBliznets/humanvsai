import sys
import csv
import nltk
import os
import re
import random
import pickle
import requests
from transformers import *
csv.field_size_limit(sys.maxsize)
from tqdm import tqdm


train_data_human=[]
train_data_chat=[]
with open('data.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        s = ','.join(row).split('|')
        if len(s[1]) > 50 and len(s[1]) < 300 and s[2]=="human":
            train_data_human.append((s[1],s[2]))
        if len(s[1]) > 50 and len(s[1]) < 300 and s[2]=="ChatGPT":
            train_data_chat.append((s[1],s[2]))
train_data_human = train_data_human[:len(train_data_chat)]
train_data_chat=train_data_chat[:len(train_data_human)]
train_data = train_data_chat+train_data_human
f = open("datachat1.csv" , "w")
writer = csv.writer(f)
for s in train_data:
  if s[1]=="human":
    writer.writerow([s[0],"human"])
  else:
    writer.writerow([s[0],"ai"])