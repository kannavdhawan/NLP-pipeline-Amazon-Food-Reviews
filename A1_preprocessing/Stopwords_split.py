import random
def stopwords_remover(stopwords_list,List_with_stopwords):
    pass
 # Removing stopwords from both positive and negative reviews
    List_without_stopwords=List_with_stopwords
    for review_type in range(0,len(List_without_stopwords)):
        for list_inner in range(0,len(List_without_stopwords[review_type])):
            for stop in stopwords_list: 
                while stop in List_without_stopwords[review_type][list_inner]:
                    List_without_stopwords[review_type][list_inner].remove(stop)
                # if stop in List_without_stopwords[review_type][list_inner]:
                #     List_without_stopwords[review_type][list_inner].remove(stop)
                # else:
                #     continue
    print(List_without_stopwords[0][0:5])
    print(List_without_stopwords[1][0:5])

    return List_without_stopwords

def train_val_test(List_with_stopwords,List_without_stopwords):
    # "" With stopwords

        # "" For positive Reviews
    
    train_data_pos=[]
    val_data_pos=[]
    test_data_pos=[]
    l_of_l_of_pos_rev=List_with_stopwords[0]
    # To track the index of list
    pos_index_list=list(range(0,len(l_of_l_of_pos_rev)))
    # indexes of train data pos reviews
    index_train_data_pos=random.sample(pos_index_list,k=int(0.8*(len(pos_index_list))))
    print(len(index_train_data_pos)) #80% of total
    remaining_indexes=list(set(pos_index_list)-set(index_train_data_pos))
    print(len(remaining_indexes)) #20% left
    # indexes of val data pos reviews
    index_val_data_pos=random.sample(remaining_indexes,int(0.5*(len(remaining_indexes))))
    print(len(index_val_data_pos)) #10% of total
    remaining_indexes=list(set(remaining_indexes)-set(index_val_data_pos))
    print(len(remaining_indexes))  #10% left
    # indexes of test data pos reviews
    index_test_data_pos=remaining_indexes # 10 % of total
    # final index lists are 
    # index_train_data_pos,index_val_data_pos,index_test_data_pos
    for i in index_train_data_pos:
        train_data_pos.append(l_of_l_of_pos_rev[i])
    for j in index_val_data_pos:
        val_data_pos.append(l_of_l_of_pos_rev[j])
    for k in index_test_data_pos:
        test_data_pos.append(l_of_l_of_pos_rev[k])
    
    print("Total instances positive reviews: ",len(pos_index_list))
    print("Size of training data positive reviews: ",len(train_data_pos))
    print("Size of validation data positive reviews: ",len(val_data_pos))
    print("Size of testing data positive reviews: ",len(test_data_pos))

     # "" With stopwords

        # "" For Negative Reviews
    
    train_data_neg=[]
    val_data_neg=[]
    test_data_neg=[]
    l_of_l_of_neg_rev=List_with_stopwords[1]
    neg_index_list=list(range(0,len(l_of_l_of_neg_rev)))
    
    index_train_data_neg=random.sample(neg_index_list,k=int(0.8*(len(neg_index_list))))
    remaining_indexes=list(set(neg_index_list)-set(index_train_data_neg))
    index_val_data_neg=random.sample(remaining_indexes,int(0.5*(len(remaining_indexes))))
    remaining_indexes=list(set(remaining_indexes)-set(index_val_data_neg))
    index_test_data_neg=remaining_indexes
    
    for i in index_train_data_neg:
        train_data_neg.append(l_of_l_of_pos_rev[i])
    for j in index_val_data_neg:
        val_data_neg.append(l_of_l_of_pos_rev[j])
    for k in index_test_data_neg:
        test_data_neg.append(l_of_l_of_pos_rev[k])
    
    print("Total instances negative reviews: ",len(neg_index_list))
    print("Size of training data negative reviews: ",len(train_data_neg))
    print("Size of validation data negative reviews: ",len(val_data_neg))
    print("Size of testing data negative reviews: ",len(test_data_neg))

     # "" Without stopwords

        # "" For positive Reviews
    
    train_data_pos_no_sw=[]
    val_data_pos_no_sw=[]
    test_data_pos_no_sw=[]
    l_of_l_of_pos_rev_no_sw=List_without_stopwords[0]
    
    # To track the index of list
    pos_index_list_no_sw=list(range(0,len(l_of_l_of_pos_rev_no_sw)))
    
    # indexes of train data pos reviews 
    index_train_data_pos_no_sw=random.sample(pos_index_list_no_sw,k=int(0.8*(len(pos_index_list_no_sw))))
    remaining_indexes=list(set(pos_index_list_no_sw)-set(index_train_data_pos_no_sw))
    
    # indexes of val data pos reviews
    index_val_data_pos_no_sw=random.sample(remaining_indexes,int(0.5*(len(remaining_indexes))))
    remaining_indexes=list(set(remaining_indexes)-set(index_val_data_pos_no_sw))
  
    # indexes of test data pos reviews
    index_test_data_pos_no_sw=remaining_indexes # 10 % of total
  
    for i in index_train_data_pos_no_sw:
        train_data_pos_no_sw.append(l_of_l_of_pos_rev_no_sw[i])
    for j in index_val_data_pos_no_sw:
        val_data_pos_no_sw.append(l_of_l_of_pos_rev_no_sw[j])
    for k in index_test_data_pos_no_sw:
        test_data_pos_no_sw.append(l_of_l_of_pos_rev_no_sw[k])
    print("------------No stopwords-----------")
    print("Total instances positive reviews: ",len(pos_index_list_no_sw))
    print("Size of training data positive reviews: ",len(train_data_pos_no_sw))
    print("Size of validation data positive reviews: ",len(val_data_pos_no_sw))
    print("Size of testing data positive reviews: ",len(test_data_pos_no_sw))

     # "" Without stopwords

        # "" For Negative Reviews
    
    train_data_neg_no_sw=[]
    val_data_neg_no_sw=[]
    test_data_neg_no_sw=[]
    l_of_l_of_neg_rev_no_sw=List_with_stopwords[1]
    neg_index_list_no_sw=list(range(0,len(l_of_l_of_neg_rev_no_sw)))
    
    index_train_data_neg_no_sw=random.sample(neg_index_list_no_sw,k=int(0.8*(len(neg_index_list_no_sw))))
    remaining_indexes=list(set(neg_index_list_no_sw)-set(index_train_data_neg_no_sw))
    index_val_data_neg_no_sw=random.sample(remaining_indexes,int(0.5*(len(remaining_indexes))))
    remaining_indexes=list(set(remaining_indexes)-set(index_val_data_neg_no_sw))
    index_test_data_neg_no_sw=remaining_indexes
    
    for i in index_train_data_neg_no_sw:
        train_data_neg_no_sw.append(l_of_l_of_pos_rev_no_sw[i])
    for j in index_val_data_neg_no_sw:
        val_data_neg_no_sw.append(l_of_l_of_pos_rev_no_sw[j])
    for k in index_test_data_neg_no_sw:
        test_data_neg_no_sw.append(l_of_l_of_pos_rev_no_sw[k])
    
    print("Total instances negative reviews: ",len(neg_index_list_no_sw))
    print("Size of training data negative reviews: ",len(train_data_neg_no_sw))
    print("Size of validation data negative reviews: ",len(val_data_neg_no_sw))
    print("Size of testing data negative reviews: ",len(test_data_neg_no_sw))
    # We have 12 list of lists.
