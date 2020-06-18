from gensim_w import load_model,most_sim
import pandas as pd
from gensim.models import Word2Vec
import os
import sys
def main(txt_path):
    with open(txt_path,'r') as f:
        list_words=f.read().splitlines()
    n=20
    for word in list_words:
        model=Word2Vec.load("data/word2vec.model")
        most_sim(model,word,n)
if __name__ == "__main__":
    main(os.sys.argv[1])