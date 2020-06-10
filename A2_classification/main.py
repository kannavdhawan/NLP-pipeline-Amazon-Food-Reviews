import numpy as np
import os 
import sys
from N_grams_generation import data_conversion
from N_grams_generation import n_grams
from N_grams_generation import n_gram_generation
from N_grams_generation import formatted_data_generation
from N_grams_generation import classify
def main(data_path):
    lists=data_conversion(data_path) # returns list of lists without labels. 
    gram_lists=n_gram_generation(lists) # list of list of grams unigram,bigram,uni+bi
    formatted_data=formatted_data_generation(gram_lists) # list of tuples of dictionary with labels.#[({'extra':True,'Time':True},1),({'No':True,'Time':True},0)]
    classify(formatted_data)
if __name__ == '__main__':
    main(sys.argv[1])