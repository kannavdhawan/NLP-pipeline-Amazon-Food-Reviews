# Dependencies
import numpy as np # Never used in Assignment1 
import os 
import sys
from Tokenizer import tokenizer
from Filter_sc import spaced_special_char_filter,spec_char_filter
from Stopwords_split import stopwords_remover,train_val_test

spec_char = ['!','"','#','%','$','&','(',')','*','+','/',':',';','<','=','>',',','@','[','\\',']','^','`','{','|','}','~','\t','\n']

stopwords_list1=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
                     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                      'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                       'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                       'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                       'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above']
stopwords_list2=["a", "about", 'other', 'some', 'such', 'no', 'nor',  'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's','below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
                          'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't","didn't"
                           'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                             'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop_words=stopwords_list1+stopwords_list2
stop_words_set=set(stop_words)
stopwords_list=list(stop_words_set)
# ---NOT A REQUIREMENT FOR ASSIGNMENT--------------------------------------------------------------------------------
# from nltk.corpus import stopwords
# stopword_list=stopwords.words('english')
# spec_char_with_digits=['!','"','#','$','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n','1','2','3','4','5','6','7','8','9','0']
#Loading dataset using pandas 
    #positive_reviews=pd.read_csv(r"F:\canada\641\Text_analytics\msci-text-analytics-s20\Reviews\pos.txt",sep="\n",header=None)#header=None to prevent making first row the header of df. 
    #negative_reviews=pd.read_csv(r"F:\canada\641\Text_analytics\msci-text-analytics-s20\Reviews\neg.txt",sep="\n",header=None)
    # print(positive_reviews.head(10))
    # print(type(positive_reviews))
    # print(negative_reviews.head(10))
    # print(type(negative_reviews))
#--------------------------------------------------------**---------------------------------------------------------------------
#Loading dataset
# def main(raw_data_path):

# def read_files(pos_path,neg_path):
def read_files(data_path):
    with open(os.path.join(data_path, 'pos.txt')) as f:
        positive_reviews = f.read()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        negative_reviews = f.read()
    # print(type(positive_reviews))
    # positive_reviews = open(pos_path, "r")
    # positive_reviews=positive_reviews.read()
    # negative_reviews=open(neg_path,"r")
    # negative_reviews=negative_reviews.read()
    return positive_reviews,negative_reviews

# print("ENTER Path for positive reviews: ")
# pos_path=input()
# print("ENTER Path for negative reviews: ")
# neg_path=input()
def main(data_path):
    files_tup=read_files(data_path)
    files=list(files_tup)
    import copy
    positive_reviews=files[0]
    negative_reviews=files[1]

    #Function calls..

    ob_tokenizer=tokenizer(positive_reviews,negative_reviews) # [tokenized_pos,tokenized_neg]     #1. calling the tokenizer function from Tokenizer.py which returns tokens.
    ob_parially_filtered=spaced_special_char_filter(spec_char,ob_tokenizer)#2. calling 2 functions from script Filter_sc.py for special character removal 
    final_tokens=spec_char_filter(spec_char,ob_parially_filtered)# [final_positive_tokens,final_negative_tokens]
    # a=[]
    # a.append(final_tokens[0])
    # a.append(final_tokens[1])
    input_tokens=copy.deepcopy(final_tokens)# Resulted in a problem of shallow copy.beacause passed by reference inside above function
    List_with_stopwords=copy.deepcopy(final_tokens)
    List_without_stopwords=stopwords_remover(stopwords_list,input_tokens)    # 3. Calling Stopwords_remover,train_val_test from Stopwords_split.py
    train_val_test(List_with_stopwords,List_without_stopwords)

if __name__ == '__main__':
    main(sys.argv[1])