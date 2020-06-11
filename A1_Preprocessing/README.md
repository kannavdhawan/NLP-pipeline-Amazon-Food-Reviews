# Assignment 1 | Preprocessing
-------------------------------------------------

# Instructions to run the code:
From the root of this repository, run `python3 A1_Preprocessing/main.py data/` .

____________________________________
# main.py: 
    Data is loaded and all the modules are called over here.
    Functions in Tokenizer, Filter_sc, Stopwords_split are called.
____________________________________
# Tokenizer.py:
    Input: pos, neg text files
    Sentence tokenization. 
    Word tokenization. 

Returns a list of tokens consisting of [tokenized_pos,tokenized_neg].
____________________________________
# Filter_sc.py:
    Input: special_char_list, ob_tokenizer which consists of [tokenized_pos,tokenized_neg] generated in previous step.
    A function "spaced_special_char_filter" which removes separated special characters which are not joined with any other word for both the positive and negative tokens.
    A function "spec_char_filter" accepting special character list and ob_parially_filtered which is the list of partially filtered tokens from the previous step and this removes the special characters which are concatinated with other characters For instance: '@#method%^' => outputs 'method'.

Returns a list final_tokens consisting of filtered positive and negative tokens without special characters. 
____________________________________

# Stopwords_split.py
    Input: stopwords_list,List_with_stopwords.
    stopwords_remover removes stopwords from both the positive and negative reviews. 
    train_val_test splits the lists [List_with_stopwords,List_without_stopwords] into
        out_sw<=stopwords list=>train_pos,val_pos,test_pos.............train_neg,val_neg,test_neg
        out_nsw<=No stopwords list=>train_pos,val_pos,test_pos.............train_neg,val_neg,test_neg
        train_sw<=train_pos+train_neg
        val_sw<=val_pos+val_neg
        test_sw<=test_pos+test_neg
        train_nsw<=train_pos_nsw+train_neg_nsw
        val_nsw<=val_pos_nsw+val_neg_nsw
        test_nsw<=test_pos_nsw+test_neg_nsw
    Data distribution: 
        80% pos and 80% neg constitute the train data, total 640000 datapoints
        10% pos and 10% neg constitute the val data, total 80000 datapoints
        10% pos and 10% neg constitute the test data, total 80000 datapoints
    Note:
        1. Reviews are Randomly sampled by indexes while index generation.
        2. Not adding [] and '' here while creating csv's. code is commented. It can be created like that.  
        3. I have created simple features of words. In classification/Assignment 2, each line/review in csv is handeled as ['word1','word2'....]
        4. In order to create no bias, I have created the equal partitions of pos,neg reviews for train,val and test for both with and without stopwords. 
