# Dependencies
import numpy as np
import pandas as pd
from Tokenizer import tokenizer
# from .Tokenizer import tokenizer
# from Preprocessing.Tokenizer import tokenizer
from Filter_sc import spaced_special_char_filter,spec_char_filter
from Stopwords_split import stopwords_remover,train_val_test
# from Tokenizer import tokenizer
import copy
spec_char = ['!','"','#','%','$','&','(',')','*','+','/',':',';','<','=','>',',','@','[','\\',']','^','`','{','|','}','~','\t','\n']
stopwords_list1=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
                     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                      'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                       'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                       'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                       'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
                          'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't","didn't"
                           'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                             'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_list2=["a", "about", 'other', 'some', 'such', 'no', 'nor',  'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's']

stop_words=stopwords_list1+stopwords_list2

stop_words_set=set(stop_words)
stopwords_list=list(stop_words_set)

# ---
# from nltk.corpus import stopwords
# stopword_list=stopwords.words('english')

#--------------------------------------------------------**---------------------------------------------------------------------
# spec_char_with_digits=['!','"','#','$','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n',
#                                         '1','2','3','4','5','6','7','8','9','0']
# Numbers concatinated with strings doesn't mean anything. 

#Loading dataset using pandas 
    #positive_reviews=pd.read_csv(r"F:\canada\641\Text_analytics\msci-text-analytics-s20\Reviews\pos.txt",sep="\n",header=None)#header=None to prevent making first row the header of df. 
    #negative_reviews=pd.read_csv(r"F:\canada\641\Text_analytics\msci-text-analytics-s20\Reviews\neg.txt",sep="\n",header=None)
    # print(positive_reviews.head(10))
    # print(type(positive_reviews))
    # print(negative_reviews.head(10))
    # print(type(negative_reviews))
#--------------------------------------------------------**---------------------------------------------------------------------
#Loading dataset
def read_files(pos_path,neg_path):
    # positive_reviews = open("F:\canada\\641\Text_analytics\msci-text-analytics-s20\Reviews\pos.txt", "r")
    positive_reviews = open(pos_path, "r")
    positive_reviews=positive_reviews.read()

    # negative_reviews=open("F:\canada\\641\Text_analytics\msci-text-analytics-s20\Reviews\\neg.txt","r")
    negative_reviews=open(neg_path,"r")
    negative_reviews=negative_reviews.read()

    return positive_reviews,negative_reviews

print("ENTER Path for positive reviews: ")
pos_path=input()
print("ENTER Path for negative reviews: ")
neg_path=input()
files_tup=read_files(pos_path,neg_path)
files=list(files_tup)

positive_reviews=files[0]
negative_reviews=files[1]


ob_tokenizer=tokenizer(positive_reviews,negative_reviews)
ob_parially_filtered=spaced_special_char_filter(spec_char,ob_tokenizer)
final_tokens=spec_char_filter(spec_char,ob_parially_filtered)

# a=[]
# a.append(final_tokens[0])
# a.append(final_tokens[1])
# problem of shallow copy.
input_tokens=copy.deepcopy(final_tokens) # with stopwords

List_without_stopwords=stopwords_remover(stopwords_list,input_tokens)

# problem of shallow copy. workaround use copy.deepcopy() instead
# List_with_stopwords=final_tokens.copy()  # beacause passed by reference inside above function 
# print(final_tokens[0][0:3])
List_with_stopwords=copy.deepcopy(final_tokens) 
train_val_test(List_with_stopwords,List_without_stopwords)
