__author__ = 'akshaykulkarni'

import pandas as pd
import numpy as np
from datetime import datetime, date, time
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


input_file_name = "train.txt"

results         = []
results2        = []
results3        = []
results4        = []
results5        = []

header          = []
counter         = 0
threshold       = 1000

with open(input_file_name) as f:
    for line in f:
        line    = line.strip("\n").strip()
        if 0 == counter:
            header  = line.split(",")
            counter = 1
        else:
            if counter <= threshold:
                results.append(line.split(","))
            elif counter >= threshold and counter <= 2*threshold:
                results2.append(line.split(","))
            elif counter >= 2*threshold and counter <= 3*threshold:
                results3.append(line.split(","))
            elif counter >= 3*threshold and counter <= 4*threshold:
                results4.append(line.split(","))
            elif counter >= 4*threshold and counter <= 5*threshold:
                results5.append(line.split(","))
            else:
                break

            counter     = counter + 1

'''
Data Processing
'''

testing                 = pd.DataFrame(data=np.asarray(results[1:]), columns=header)
testing.click           = testing.click.astype(int)
testing['timestamp']    = testing['hour'].map(lambda x: int(datetime.strptime(x[4:6] + "/" + x[2:4] + "/" + x[0:2] + " " + x[6:8], "%d/%m/%y %H").strftime("%s")))
trial_text              = header[3:-1]
df                      = pd.DataFrame()

for elem in trial_text:
    interim = pd.get_dummies(testing[elem])
    interim.rename(columns=lambda x: elem + "_" + x, inplace=True)
    df      = df.join(interim)

for ix, value in enumerate(testing.ix[:,3:-1].values):
    mydict = {}
    indy = []
    indy.append(ix)
    for j, title in enumerate(value):
        mydict[header[j+3] + "_" + title] = 1
    df2 = pd.DataFrame(mydict, index = indy)
    df = df.append(df2)

df.fillna(0, inplace=True)
df['timestamp'] = testing['timestamp']

'''
Model-Fitting
'''

Y                               = testing.click
xtrain, xtest, ytrain, ytest    = train_test_split(df,Y)
clf = MultinomialNB().fit(xtrain, ytrain)
print "Training accuracy: %0.2f%%" % (100 * clf.score(xtrain, ytrain))
print "Test accuracy: %0.2f%%" % (100 * clf.score(xtest, ytest))


prob = clf.predict_proba(xtest)
plt.hist(prob[:,1])
plt.ylabel("Count")
plt.xlabel("Probability")
plt.title("Naive Bayes Probability Histogram")
plt.savefig('naive_bayes.png')

clf2            = LogisticRegression().fit(xtrain, ytrain)
print "Training accuracy: %0.2f%%" % (100 * clf2.score(xtrain, ytrain))
print "Test accuracy: %0.2f%%" % (100 * clf2.score(xtest, ytest))

prob = clf2.predict_proba(xtest)
plt.hist(prob[:,1])
plt.ylabel("Count")
plt.xlabel("Probability")
plt.title("Logistic Regression Probability Histogram")
plt.savefig("logistic_reg.png")






