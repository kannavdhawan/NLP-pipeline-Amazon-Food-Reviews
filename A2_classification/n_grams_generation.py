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