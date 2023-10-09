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
with open('chatgpt_paraphrases.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row[0])>100 and len(row[0])<200 and row[2]=='sentence':
            print(row[0])
            while row[0][0] == "'" or row[0][0] == '"':
               row[0]=row[0][1:]
            while row[0][-1] == "'" or row[0][-1] == '"':
               row[0]=row[0][:-1]
            print(row[0])
            train_data_human.append((row[0], "human"))
            
            row[1]=row[1][1:]
            row[1]=row[1][1:]
            
            while row[1][0] == "'" or row[1][0] == '"':
               row[1]=row[1][1:]
            i1 = row[1].find("',")
            i2 = row[1].find('",')
            if i1 == -1:
               i1 = len(row[1])
            if i2 == -1:
               i2 = len(row[1])
            print(i1)
            print(i2)
            train_data_chat.append((row[1][: min((i1, i2))], "ai"))

train_data = train_data_chat+train_data_human
f = open("datachat3.csv" , "w")
writer = csv.writer(f)
for s in train_data:
  if s[1]=="human":
    writer.writerow([s[0],"human"])
  else:
    writer.writerow([s[0],"ai"])
        # if len(s[1]) > 100 and len(s[1]) < 200 and s[2]=="human":
        #     train_data_human.append((s[1],s[2]))
        # if len(s[1]) > 100 and len(s[1]) < 200 and s[2]=="ChatGPT":
        #     train_data_chat.append((s[1],s[2]))
# train_data_chat=train_data_chat[:len(train_data_human)]
# train_data = train_data_chat+train_data_human
# f = open("datachat1.csv" , "w")
# writer = csv.writer(f)
# for s in train_data:
#   if s[1]=="human":
#     writer.writerow([s[0],"human"])
#   else:
#     writer.writerow([s[0],"ai"])