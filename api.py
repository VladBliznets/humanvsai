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


test=[]
with open('quora.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        s = ','.join(row)
        if (len(s) > 50 and len(s)<100):
            test.append(s)

f = open("Pegas.csv" , "w")
writer = csv.writer(f)
for s in test:
  writer.writerow([s,"human"])
     
print(len(test))
random.shuffle(test)
print(test[:1])

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=1, num_beams=1):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

k = 0
for req in tqdm(test):
  try:
    writer.writerow([get_paraphrased_sentences(model, tokenizer, req), "ai"])
  except:
    k=k+1

print(k)


