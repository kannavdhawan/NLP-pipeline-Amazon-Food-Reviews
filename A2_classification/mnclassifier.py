import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

train=pd.read_csv("train_sw.csv",sep=';',header=None)  
train=train[0].str.split(",",expand=True)
print(train.iloc[:,1:].head())
print(train.info())



clf=MultinomialNB()
clf.fit(train.iloc[1:], train[0])
print(clf.score(train.iloc[1:], train[0]))
