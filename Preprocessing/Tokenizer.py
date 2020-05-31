import numpy as np
import pandas as pd
def tokenizer(pos,neg):
    tokenized_pos=[]
    tokenized_neg=[]
    tokens=[]
    pos=pos.lower()
    neg=neg.lower()
# -------splitting by \n -- sentence tokenization-------
    positive_reviews_split=pos.splitlines() # returns a list
    negative_reviews_split=neg.splitlines()
    # print(positive_reviews_split[1:3])
#----word tokenization----
    for i in positive_reviews_split:
        pass
        tokenized_pos.append(i.split())
    print("Tokenized pos subset-->")
    print(tokenized_pos[0:3]) 

    tokens.append(tokenized_pos)

    for j in negative_reviews_split:
        pass
        tokenized_neg.append(j.split())
    print("Tokenized neg subset-->")
    print(tokenized_neg[0:3])
    tokens.append(tokenized_neg)

    return tokens



