classifiers = ["Logistic", "LinearSVC", "NuSVC", "BernoulliNB", "MultinomialNB", "SVC"]
numbers_of_words = [10, 50, 100, 200, 400, 500, 800, 1000]
coefs_of_ngrams = [0,10,100]
dataset = "dc2"
for a in numbers_of_words:
    for b in classifiers:
        for c in coefs_of_ngrams:
            print((str(a)+"_"+str(c)+"_"+str(b)+"_"+str(dataset)+'", "'), end='')
