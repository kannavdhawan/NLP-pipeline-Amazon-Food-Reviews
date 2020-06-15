
def data_formatting(input_gram,length):
    temp=[]
    for i in range(len(input_gram)):
        x=1
        y=0
        temp_dictionary=dict()
        for j in range(len(input_gram[i])):
            temp_dictionary[input_gram[i][j]]=True # making it always true for every word to preserve all the details.
        if i<length:
            temp.append((temp_dictionary,x))
        else:
            temp.append((temp_dictionary,y))
    return temp
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