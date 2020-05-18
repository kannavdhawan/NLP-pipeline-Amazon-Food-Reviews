# Dependencies
import numpy as np
import pandas as pd
from Tokenizer import tokenizer
from Filter_sc import spaced_special_char_filter,spec_char_filter
from Stopwords_split import stopwords_remover
spec_char = ['!','"','#','$','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n',',']

stopwords_list1=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
                     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                      'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                       'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
                       'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                       'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                         'more', 'most', 'other', 'some', 'such', 'no', 'nor',  'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
                          'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                           'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                             'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_list2=["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", 
"aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
"can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't",
 "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",
  "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
   "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
    "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",  "now", "o", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's",
     ''"should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their",
      "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
       "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
         "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", 
         "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", 
         "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]

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
positive_reviews = open("F:\canada\\641\Text_analytics\msci-text-analytics-s20\Reviews\pos.txt", "r")
positive_reviews=positive_reviews.read()

negative_reviews=open("F:\canada\\641\Text_analytics\msci-text-analytics-s20\Reviews\\neg.txt","r")
negative_reviews=negative_reviews.read()




ob_tokenizer=tokenizer(positive_reviews,negative_reviews)
ob_parially_filtered=spaced_special_char_filter(spec_char,ob_tokenizer)
final_tokens=spec_char_filter(spec_char,ob_parially_filtered)
List_with_stopwords=final_tokens
List_without_stopwords=stopwords_remover(stopwords_list,List_with_stopwords)
# train_val_test(List_with_stopwords,List_without_stopwords)