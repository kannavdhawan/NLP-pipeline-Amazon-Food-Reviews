import re 

import sys

import numpy as np

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score
def file_read(file_set):
    
    data = []
    
    string = open(file_set, 'r')
    
    for char in string:
        print(char)
        # break
        char_new = re.sub(r"\[|\]|\,|\'|\"|\n", '', char)
        # char_new=char_new.replace(',',' ',len(char)-1)        
        if char_new != " ":
        
            data.append(char_new)
    
    string.close()
    
    return data    
import os 
train_all_tokens = file_read('data/val_nsw.csv')
print(train_all_tokens[0:5])

def count_vect(train_tokens, val_tokens, test_tokens, x,y):
    
    n_range = (x,y)
    
    vect = CountVectorizer(ngram_range = n_range)
    
    X_train = vect.fit_transform(train_tokens)
    
    X_val = vect.transform(val_tokens)
    
    X_test = vect.transform(test_tokens)
    print(len(list(vect.get_feature_names())))

    return X_train, X_val, X_test
X_all_train_unigram, X_all_val_unigram, X_all_test_unigram = count_vect(train_all_tokens, train_all_tokens, train_all_tokens, 1, 1)
print(X_all_test_unigram.toarray())


# Multinomial Naive Bias Classifier

def MNB_classifier(X_train, X_val, y_train, y_val):
    
    alpha = [0, 0.1, 0.5, 1, 10, 20, 50, 100]
    
    for value in alpha:
        
        clf = MultinomialNB(alpha = value)
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        
        score = accuracy_score(y_val, y_pred)
        
        percent_score = 100 * score
        
        print("For alpha = {}  prediction accuracy is {}%".format(value, percent_score))
        
    return percent_score
        
    