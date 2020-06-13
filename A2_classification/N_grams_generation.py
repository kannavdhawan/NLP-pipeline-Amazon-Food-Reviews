import pandas as pd
import numpy as np
import nltk
import os
import random
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
def data_conversion(data_path):
    train_sw=pd.read_csv(os.path.join(data_path, "train_sw.csv"),sep=';',header=None,names=['Text'])
    train_nsw=pd.read_csv(os.path.join(data_path, "train_nsw.csv"),sep=';',header=None,names=['Text'])
    val_sw=pd.read_csv(os.path.join(data_path, "val_sw.csv"),sep=';',header=None,names=['Text'])
    val_nsw=pd.read_csv(os.path.join(data_path, "val_nsw.csv"),sep=';',header=None,names=['Text'])
    test_sw=pd.read_csv(os.path.join(data_path, "test_sw.csv"),sep=';',header=None,names=['Text'])
    test_nsw=pd.read_csv(os.path.join(data_path, "test_nsw.csv"),sep=';',header=None,names=['Text'])
    print(train_sw.head())
    print(test_sw.info())
    print("train sw :")
    print(train_sw.head(3))
    
    #-------------FOLLOWING CODE CAN BE USED IF CSV CONTAINS STRINGS IN THE FORM OF '' -----------------------.
        
    # for i in range(len(train_sw['Text'])):
    #     train_sw_list.append(list(eval(train_sw.iloc[i,0])))
    # print(train_sw_list[0:5])
    # train_sw_list.append(eval('['+train_sw.iloc[20,0]+']'))
    # print(train_sw_list)

    # Converting into list of list without labels
    
    # Extracting training data into list of list
    
    temp_train_sw=[]
    train_sw_list=[]
    for i in range(len(train_sw)):
        temp_train_sw=(train_sw.iloc[i,0])[:-1].split(',') # -1 because of extra comma added using csv creation.
        train_sw_list.append(temp_train_sw)
    print(train_sw_list[0:10])
    print(len(train_sw_list))

    temp_train_nsw=[]
    train_nsw_list=[]
    for i in range(len(train_nsw)):
        temp_train_nsw=(train_nsw.iloc[i,0])[:-1].split(',')
        train_nsw_list.append(temp_train_nsw)
    print(train_nsw_list[0:10])
    print(len(train_nsw_list))

    # Extarcting validation data into list of list 
    temp_val_sw=[]
    val_sw_list=[]
    for i in range(len(val_sw)):
        temp_val_sw=(val_sw.iloc[i,0])[:-1].split(',')
        val_sw_list.append(temp_val_sw)
    print(val_sw_list[0:10])
    print(len(val_sw_list))

    temp_val_nsw=[]
    val_nsw_list=[]
    for i in range(len(val_nsw)):
        temp_val_nsw=(val_nsw.iloc[i,0])[:-1].split(',')
        val_nsw_list.append(temp_val_nsw)
    print(val_nsw_list[0:10])
    print(len(val_nsw_list))

    # Extarcting Testing data into list of list 
    temp_test_sw=[]
    test_sw_list=[]
    for i in range(len(test_sw)):
        temp_test_sw=(test_sw.iloc[i,0])[:-1].split(',')
        test_sw_list.append(temp_test_sw)
    print(test_sw_list[0:10])
    print(len(test_sw_list))

    temp_test_nsw=[]
    test_nsw_list=[]
    for i in range(len(test_nsw)):
        temp_test_nsw=(test_nsw.iloc[i,0])[:-1].split(',')
        test_nsw_list.append(temp_test_nsw)
    print(test_nsw_list[0:10])
    print(len(test_nsw_list))
    
    return train_sw_list,train_nsw_list,val_sw_list,val_nsw_list,test_sw_list,test_nsw_list #list of lists 

# GENERATING N GRAMS

def n_grams(input_set,n):
    gram_list=[]
    for i in range(len(input_set)):
        gram_list.append([list(input_set[i][j:j+n]) for j in range(len(input_set[i])-(n-1))])
    # print(gram_list[0:2])
    for outer in range(len(gram_list)):
        for inner in range(len(gram_list[outer])):
            gram_list[outer][inner]=" ".join(gram_list[outer][inner]) #The join() method takes all items in an iterable and joins them into one string.
    return gram_list
def n_gram_generation(input_lists):
    train_sw_list=input_lists[0]
    train_nsw_list=input_lists[1]
    val_sw_list=input_lists[2]
    val_nsw_list=input_lists[3]
    test_sw_list=input_lists[4]
    test_nsw_list=input_lists[5]
#unigrams
    u=1                                                                # stopwords set-1
    unigram_train_sw=n_grams(train_sw_list,u)
    unigram_val_sw=n_grams(val_sw_list,u)
    unigram_test_sw=n_grams(test_sw_list,u)
    print("set 1: test ",len(unigram_test_sw))
    print("testing some unigrams :", unigram_test_sw[0:2])
                                                                    # No stopwords set-2
    unigram_train_nsw=n_grams(train_nsw_list,u)
    unigram_val_nsw=n_grams(val_nsw_list,u)
    unigram_test_nsw=n_grams(test_nsw_list,u)
    print("set 2: train ",len(unigram_train_nsw))

#bigrams

    u=2                                                                # stopwords set-3 
    bigrams_train_sw=n_grams(train_sw_list,u)
    bigrams_val_sw=n_grams(val_sw_list,u)
    bigrams_test_sw=n_grams(test_sw_list,u)
    print("set 3: train ",len(bigrams_train_sw))
    print("testing some unigrams :", bigrams_test_sw[0:2])

                                                                    # No stopwords set-4
    bigrams_train_nsw=n_grams(train_nsw_list,u)
    bigrams_val_nsw=n_grams(val_nsw_list,u)
    bigrams_test_nsw=n_grams(test_nsw_list,u)
    print("set 4: train ",len(bigrams_train_nsw))

#unigram+bigrams 

    # stopwords set-5
    ub_train_sw=[]
    for i in range(len(unigram_train_sw)):
        ub_train_sw.append(unigram_train_sw[i]+bigrams_train_sw[i])

    ub_val_sw=[]
    for i in range(len(unigram_val_sw)):
        ub_val_sw.append(unigram_val_sw[i]+bigrams_val_sw[i])

    ub_test_sw=[]
    for i in range(len(unigram_test_sw)):
        ub_test_sw.append(unigram_test_sw[i]+bigrams_test_sw[i])
    print("testing some unigrams+bigrams :", ub_test_sw[0:2])
#No stopwords set-6
    ub_train_nsw=[]
    for i in range(len(unigram_train_nsw)):
        ub_train_nsw.append(unigram_train_nsw[i]+bigrams_train_nsw[i])

    ub_val_nsw=[]
    for i in range(len(unigram_val_nsw)):
        ub_val_nsw.append(unigram_val_nsw[i]+bigrams_val_nsw[i])

    ub_test_nsw=[]
    for i in range(len(unigram_test_nsw)):
        ub_test_nsw.append(unigram_test_nsw[i]+bigrams_test_nsw[i])
    print("testing uni+bi::")
    print(ub_test_nsw[:5])
    return unigram_train_sw,unigram_val_sw,unigram_test_sw,unigram_train_nsw,unigram_val_nsw,unigram_test_nsw,bigrams_train_sw,bigrams_val_sw,bigrams_test_sw,bigrams_train_nsw,bigrams_val_nsw,bigrams_test_nsw,ub_train_sw,ub_val_sw,ub_test_sw,ub_train_nsw,ub_val_nsw,ub_test_nsw
x=1
y=0

# def dic(j):
#     return dict([(word, True) for word in j]) # returns dictionary
# def data_formatting(input,label_size):
#     temp=[]
#     for i in range(len(input)):
#         if i<=label_size:
#             temp.append((dict([j, True]),1) for j in input[i]) #format-->  [({},1)]
#         else:
#             temp.append((dict([(j, True)]),0) for j in input[i]) #format-->  [({},0)]
#     return temp


# :param labeled_featuresets: A list of ``(featureset, label)``
#             where each ``featureset`` is a dict mapping strings to either
#             numbers, booleans or strings.

def data_formatting(input_gram,length):
    temp=[]
    for i in range(len(input_gram)):
        temp_dictionary=dict()
        for j in range(len(input_gram[i])):
            temp_dictionary[input_gram[i][j]]=True # making it always true for every word to preserve all the details.
        if i<length:
            temp.append((temp_dictionary,x))
        else:
            temp.append((temp_dictionary,y))
    return temp

#[({"hello":True,},1),()]
def formatted_data_generation(gram_list):
    unigram_train_sw=gram_list[0]
    unigram_val_sw=gram_list[1]
    unigram_test_sw=gram_list[2]
    unigram_train_nsw=gram_list[3]
    unigram_val_nsw=gram_list[4]
    unigram_test_nsw=gram_list[5]
    
    bigrams_train_sw=gram_list[6]
    bigrams_val_sw=gram_list[7]
    bigrams_test_sw=gram_list[8]
    
    bigrams_train_nsw=gram_list[9]
    bigrams_val_nsw=gram_list[10]
    bigrams_test_nsw=gram_list[11]
    
    ub_train_sw=gram_list[12]
    ub_val_sw=gram_list[13]
    ub_test_sw=gram_list[14]
    
    ub_train_nsw=gram_list[15]
    ub_val_nsw=gram_list[16]
    ub_test_nsw=gram_list[17]
    x=2
    unigram_train_sw_final=data_formatting(unigram_train_sw,len(unigram_train_sw)/2)
    unigram_val_sw_final=data_formatting(unigram_val_sw,len(unigram_val_sw)/2)
    unigram_test_sw_final=data_formatting(unigram_test_sw,len(unigram_test_sw)/2)
    unigram_train_nsw_final=data_formatting(unigram_train_nsw,len(unigram_train_nsw)/2)
    unigram_val_nsw_final=data_formatting(unigram_val_nsw,len(unigram_val_nsw)/2)
    unigram_test_nsw_final=data_formatting(unigram_test_nsw,len(unigram_test_nsw)/2)
    bigrams_train_sw_final=data_formatting(bigrams_train_sw,len(bigrams_train_sw)/2)
    bigrams_val_sw_final=data_formatting(bigrams_val_sw,len(bigrams_val_sw)/2)
    bigrams_test_sw_final=data_formatting(bigrams_test_sw,len(bigrams_test_sw)/2)
    bigrams_train_nsw_final=data_formatting(bigrams_train_nsw,len(bigrams_train_nsw)/2)
    bigrams_val_nsw_final=data_formatting(bigrams_val_nsw,len(bigrams_val_nsw)/2)
    bigrams_test_nsw_final=data_formatting(bigrams_test_nsw,len(bigrams_test_nsw)/2)
    ub_train_sw_final=data_formatting(ub_train_sw,len(ub_train_sw)/2)
    ub_val_sw_final=data_formatting(ub_val_sw,len(ub_val_sw)/2)
    ub_test_sw_final=data_formatting(ub_test_sw,len(ub_test_sw)/2)
    ub_train_nsw_final=data_formatting(ub_train_nsw,len(ub_train_nsw)/2)
    ub_val_nsw_final=data_formatting(ub_val_nsw,len(ub_val_nsw)/2)
    ub_test_nsw_final=data_formatting(ub_test_nsw,len(ub_test_nsw)/2)
    return unigram_train_sw_final,unigram_val_sw_final,unigram_test_sw_final,unigram_train_nsw_final,unigram_val_nsw_final,unigram_test_nsw_final,bigrams_train_sw_final,bigrams_val_sw_final,bigrams_test_sw_final,bigrams_train_nsw_final,bigrams_val_nsw_final,bigrams_test_nsw_final,ub_train_sw_final,ub_val_sw_final,ub_test_sw_final,ub_train_nsw_final,ub_val_nsw_final,ub_test_nsw_final
# classification 
def classify(formatted_data):
    unigram_train_sw_final=formatted_data[0]
    unigram_val_sw_final=formatted_data[1]
    unigram_test_sw_final=formatted_data[2]
    unigram_train_nsw_final=formatted_data[3]
    unigram_val_nsw_final=formatted_data[4]
    unigram_test_nsw_final=formatted_data[5]
    bigrams_train_sw_final=formatted_data[6]
    bigrams_val_sw_final=formatted_data[7]
    bigrams_test_sw_final=formatted_data[8]
    bigrams_train_nsw_final=formatted_data[9]
    bigrams_val_nsw_final=formatted_data[10]
    bigrams_test_nsw_final=formatted_data[11]
    ub_train_sw_final=formatted_data[12]
    ub_val_sw_final=formatted_data[13]
    ub_test_sw_final=formatted_data[14]
    ub_train_nsw_final=formatted_data[15]
    ub_val_nsw_final=formatted_data[16]
    ub_test_nsw_final=formatted_data[17]
    # list of sets

    unigrams_sw=[unigram_train_sw_final,unigram_val_sw_final,unigram_test_sw_final]
    unigrams_nsw=[unigram_train_nsw_final,unigram_val_nsw_final,unigram_test_nsw_final]
    bigrams_sw=[bigrams_train_sw_final,bigrams_val_sw_final,bigrams_test_sw_final]
    bigrams_nsw=[bigrams_train_nsw_final,bigrams_val_nsw_final,bigrams_test_nsw_final]
    ub_sw=[ub_train_sw_final,ub_val_sw_final,ub_test_sw_final]
    ub_nsw=[ub_train_nsw_final,ub_val_nsw_final,ub_test_nsw_final]
    #reshuffling already shuffled data in assignment 1
    for i in unigrams_sw:
        random.shuffle(i)
    for i in unigrams_nsw:
        random.shuffle(i)
    for i in bigrams_sw:
        random.shuffle(i)
    for i in bigrams_nsw:
        random.shuffle(i)
    for i in ub_sw:
        random.shuffle(i)
    for i in ub_nsw:
        random.shuffle(i)
    # Alpha::::::::::
    # alpha suppresses the effect of rare words. For instance, if there is only 1 spam email out of 20 emails
    # in the training set, then without having significant additive smoothing, model will classify the test data/emails 
    # as spam if that word is there in any of the email.

    #unigrams stopwords
    alpha_vals=[0.1,0.4,0.5,1.0,1.5]

    print("---------------------------------unigram stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i,fit_prior=True, class_prior=None))
        MNB_classifier.train(unigrams_sw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, unigrams_sw[1])
        print("Unigram sw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("Unigrams sw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(unigrams_sw[0])
    print("Unigrams sw test accuracy=",nltk.classify.accuracy(MNB_classifier,unigrams_sw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")

    print("---------------------------------unigram No stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i, fit_prior=True, class_prior=None))
        MNB_classifier.train(unigrams_nsw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, unigrams_nsw[1])
        print("Unigram nsw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("Unigrams nsw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(unigrams_nsw[0])
    print("Unigrams nsw test accuracy=",nltk.classify.accuracy(MNB_classifier,unigrams_nsw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")

    print("---------------------------------Bigram stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i, fit_prior=True, class_prior=None))
        MNB_classifier.train(bigrams_sw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, bigrams_sw[1])
        print("Bigram sw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("Bigrams sw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(bigrams_sw[0])
    print("Bigrams sw test accuracy=",nltk.classify.accuracy(MNB_classifier,bigrams_sw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")

    print("---------------------------------Bigram No stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i, fit_prior=True, class_prior=None))
        MNB_classifier.train(bigrams_nsw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, bigrams_nsw[1])
        print("Bigram nsw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("Bigrams nsw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(bigrams_nsw[0])
    print("Bigrams nsw test accuracy=",nltk.classify.accuracy(MNB_classifier,bigrams_nsw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")


    print("---------------------------------unigram+bigram stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i, fit_prior=True, class_prior=None))
        MNB_classifier.train(ub_sw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, ub_sw[1])
        print("unigram+bigram sw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("unigram+bigram sw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(ub_sw[0])
    print("unigram+bigram sw test accuracy=",nltk.classify.accuracy(MNB_classifier,ub_sw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")


    print("---------------------------------unigram+bigram No stopwords----------------------------------------------")
    val_a=[]
    for i in alpha_vals:
        MNB_classifier = SklearnClassifier(MultinomialNB(alpha=i, fit_prior=True, class_prior=None))
        MNB_classifier.train(ub_nsw[0])
        val_acc=nltk.classify.accuracy(MNB_classifier, ub_nsw[1])
        print("unigram+bigram nsw Val acc at alpha=",i," is ",val_acc)
        val_a.append(val_acc)
    print("\n")
    print("unigram+bigram nsw val Best accuracy=",max(val_a)," at alpha=",alpha_vals[val_a.index(max(val_a))])

    MNB_classifier = SklearnClassifier(MultinomialNB(alpha=alpha_vals[val_a.index(max(val_a))], fit_prior=True, class_prior=None))
    MNB_classifier.train(ub_nsw[0])
    print("unigram+bigram nsw test accuracy=",nltk.classify.accuracy(MNB_classifier,ub_nsw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")