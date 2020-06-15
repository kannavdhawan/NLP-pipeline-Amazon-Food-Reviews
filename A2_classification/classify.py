import random
import pandas as pd
import numpy as np
import nltk
import os
import random
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
# classification 


# <!-- SklearnClassifier using Nltk -->
# <!-- 
 #1. uses zip for format provided i.e [({"hello":True},1),(),()] 
 #2. calls the dict vectorizer through train.
 #3. Then we feed it with sklearn's mnb classifier -->
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
    mnb_uni=MNB_classifier.train(unigrams_sw[0])
    # pickle.dump(mnb_uni,open(os.path.join("data/","mnb_uni.pkl"),"wb"))

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
    mnb_uni_ns=MNB_classifier.train(unigrams_nsw[0])
    # pickle.dump(mnb_uni_ns,open(os.path.join("data/","mnb_uni_ns.pkl"),"wb"))

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
    mnb_bi=MNB_classifier.train(bigrams_sw[0])
    # pickle.dump(mnb_bi,open(os.path.join("data/","mnb_bi.pkl"),"wb"))
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
    mnb_bi_ns=MNB_classifier.train(bigrams_nsw[0])
    # pickle.dump(mnb_bi_ns,open(os.path.join("data/","mnb_bi_ns.pkl"),"wb"))
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
    mnb_uni_bi=MNB_classifier.train(ub_sw[0])
    # pickle.dump(mnb_uni_bi,open(os.path.join("data/","mnb_uni_bi.pkl"),"wb"))
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
    mnb_uni_bi_ns=MNB_classifier.train(ub_nsw[0])
    # pickle.dump(mnb_uni_bi_ns,open(os.path.join("data/","mnb_uni_bi_ns.pkl"),"wb"))
    print("unigram+bigram nsw test accuracy=",nltk.classify.accuracy(MNB_classifier,ub_nsw[2])," at best value of alpha")
    print("------------------------------------------------------------------------------------------------")