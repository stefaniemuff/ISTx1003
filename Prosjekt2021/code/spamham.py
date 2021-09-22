#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:00:43 2021

@author: steffi
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split, cross_val_score

# plotting
import matplotlib.pyplot as plt

#df = pd.read_csv('~/Teaching/ISTT1003/2021/Prosjekt2021/data/spam.csv')


df = pd.read_csv('~/Teaching/ISTT1003/2021/Prosjekt2021/data/SMSSpamCollection.txt', delimiter='\t',header=None)

# Rename the columns into response y and text
df = df.rename(columns={0: 'y', 1: 'text'})

df.replace(('spam', 'ham'), (1, 0), inplace=True)

# Først del dataene i en trenings og testsett (60-40%)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['text'],df['y'],test_size=0.4)

# Og så del testsettet igjen i en test og en valideringsset
X_test_raw, X_val_raw, y_test, y_val = train_test_split(X_test_raw,y_test,test_size=0.5)


vect = CountVectorizer()
vect.fit(X_train_raw)
x_train = vect.transform(X_train_raw)
x_test = vect.transform(X_test_raw)



# ham_words = ''
# spam_words = ''
# spam = df[df.y == 1]
# ham = df[df.y == 0]


 
# Egentlig har 
lr = LogisticRegression(penalty='l2')
lr.fit(x_train,y_train)

## Score for the training set (how good is the spam filter?)

lrscore = lr.score(x_train,y_train)
print('Logistic regression score ',lrscore)

### Score for the test set

lrscore = lr.score(x_test,y_test)
print('Logistic regression score ',lrscore)


## Probability of spam/ham for the test set:

test_prob = lr.predict_proba(X=x_test)


#### Test code ###########

# Spesifiser verdi for cutoff (kan spilles med dette)
cutoff = 0.5

# Prediker sannsynlighet for spam for testsett
test_pred = test_prob[:,[1]]
plt.hist(test_pred,histtype='bar',rwidth=0.9)

# klassifiser som seier for spiller 1 hvis sannsynligheten for at spiller 1 vant er over 0.5
y_testpred = np.where(test_pred > cutoff, 1, 0)

# Finn andel korrekte klassifikasjoner
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=y_testpred),
      "Feilrate:", 1-accuracy_score(y_true=y_test, y_pred=y_testpred))
 
## Prøv andere cut-off verdier og se om du kan forbedre accuracy scoren:

# cutoff = ... 
# kopief coden fra oppover


























