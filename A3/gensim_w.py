import pandas as pd
from gensim.models import Word2Vec
import os
import sys
def load_data(data_path):
    '''
uncomment below till ln 16 if training model using {out_sw.csv/out_nsw}.
'''
    dataset_df=pd.read_csv(os.path.join(data_path,'out_sw.csv'),sep=';',header=None,names=['Reviews'])
    print("DataFrame check: ",dataset_df.head(3))
    formatted_dataset=[]

    for i in range(len(dataset_df)):
        temp=(dataset_df.iloc[i,0])[:-1].split(',')
        formatted_dataset.append(temp)
    print("LofL form df: ",formatted_dataset[0:5])
    '''
uncomment below code till __ if training model using {pos.txt+neg.txt}.
'''


    return formatted_dataset

def save_model(formatted_dataset):
    print("Training...")
    w2v=Word2Vec(sentences=formatted_dataset,min_count=1,window=5, size=250,workers=4) #sample=e-5, alpha=0.01,min_alpha=0.0001
    w2v.save("data/word2vec.model")
    return "data/word2vec.model"



def most_sim(model,word,n):
    for i in range(n):
        print(model.wv.most_similar(positive=[word], topn=n)[i])

def load_model(model_path):

    model=Word2Vec.load(model_path)
    print("---------------Good: Most similar words---------------")
    most_sim(model,"good",20)
    print("---------------Bad: Most similar words---------------")
    most_sim(model,"bad",20)