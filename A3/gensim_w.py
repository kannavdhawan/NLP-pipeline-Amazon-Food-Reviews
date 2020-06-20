import pandas as pd
from gensim.models import Word2Vec
import os
import sys
def load_data(data_path):
    '''
uncomment below till ln 16 if training model using {out_sw.csv/out_nsw}.
'''
    # dataset_df=pd.read_csv(os.path.join(data_path,'out_nsw.csv'),sep=';',header=None,names=['Reviews'])
    # print("DataFrame check: ",dataset_df.head(3))
    
    # formatted_dataset=[]
    # for i in range(len(dataset_df)):
    #     temp=(dataset_df.iloc[i,0])[:-1].split(',')
    #     formatted_dataset.append(temp)
    # print("LofL form df: ",formatted_dataset[0:5])
    '''
uncomment below code till line 60 if training model using {pos.txt+neg.txt}.
'''
    spec_char = ['!','"','#','%','$','&','(',')','*','+','/',':',';','<','=','>',',','@','[','\\',']','^','`','{','|','}','~','\t','\n','.','-','1','2','3','4','5','6','7','8','9','0','?']
    with open(os.path.join(data_path,'pos.txt'),'r') as f:
        pos=f.read().splitlines()   #['This product is best','Its so great']
    with open(os.path.join(data_path,'neg.txt'),'r') as f:
        neg=f.read().splitlines()   #['This product is worst','Its so bad']
    all_lines=pos+neg
    print("Raw data from txt combined: ",all_lines[0:3])

    all_lines=[line.split() for line in all_lines]
    print("List of list txt: ",all_lines[0:5])
    print("size:",len(all_lines))
    dataset=all_lines       #[['This', 'Product'],['Its','so']]
    #------ Filter "Explicitly spaced" special characters on positive reviews -------
    flist1=[]
    for list1 in dataset:
        innerlist=[]
        for word in list1:
            if word not in spec_char:
                innerlist.append(word)
            else:
                continue
        flist1.append(innerlist)
    print("Partially removed spec char | flist1 : ",flist1[0:10])
# Removing symbols concatinated with other strings
    formatted_dataset=[]
    for innerlist in flist1: # ['','',']
        new_list=[] # Remaking above list  
        for word in innerlist:
            val=[] # temp list for each word
            for character in word:
                if character in spec_char:
                    continue
                else:
                    val.append(character)
            string=""
            string=string.join(val)
            new_list.append(string)
        formatted_dataset.append(new_list)

    print(" final List of list: ",formatted_dataset[0:10])

    return formatted_dataset

def save_model(formatted_dataset):
    print("Training...")
    w2v=Word2Vec(sentences=formatted_dataset,min_count=20, size=200,window=5,workers=4) #sample=e-5, alpha=0.01,min_alpha=0.0001
    w2v.save("data/word2vec.model")
    return "data/word2vec.model"


def most_sim(model,word,n):
    print("\n\n---------------",word,": Most similar words---------------")
    for i in range(n):
        print(model.most_similar(positive=[word], topn=n)[i])

def load_model(model_path):

    model=Word2Vec.load(model_path)    
    most_sim(model,"good",20)
    most_sim(model,"bad",20)