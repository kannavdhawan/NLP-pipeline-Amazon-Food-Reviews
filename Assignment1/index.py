# Dependencies
import numpy as np
import pandas as pd
from Tokenizer import tokenizer
from Filter_sc import spaced_special_char_filter,spec_char_filter
spec_char = ['!','"','#','$','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','\n',',']

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
