import csv
import sys

csv.field_size_limit(sys.maxsize)
with open('quora.csv', 'r',encoding="utf-8") as file:
    reader = csv.reader(file)
    f = open("quora_output.csv" , "w")
    writer = csv.writer(f)
    for row in reader:
        if len(row[0]) < 1000:
            writer.writerow([row[0], "human"])
