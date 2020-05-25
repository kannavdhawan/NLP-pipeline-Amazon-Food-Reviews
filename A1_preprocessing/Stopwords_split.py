import random
import numpy as np
import copy

# random.seed(1332)

def stopwords_remover(stopwords_list,List_with_stopwords):
    pass
 # Removing stopwords from both positive and negative reviews
    # List_without_stopwords=List_with_stopwords # pass by reference 
    List_without_stopwords=List_with_stopwords.copy()
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
    print("##########################Testing List with stopwords######################")
    print(List_with_stopwords[0][0:3])
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
    print("------stopword pos reviews------------------")
    print("------Checking correctness of train data----------")
    print(np.array(train_data_pos).shape)
    print(train_data_pos[0:10])



    

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
        train_data_neg.append(l_of_l_of_neg_rev[i])
    for j in index_val_data_neg:
        val_data_neg.append(l_of_l_of_neg_rev[j])
    for k in index_test_data_neg:
        test_data_neg.append(l_of_l_of_neg_rev[k])
    
    print("Total instances negative reviews: ",len(neg_index_list))
    print("Size of training data negative reviews: ",len(train_data_neg))
    print("Size of validation data negative reviews: ",len(val_data_neg))
    print("Size of testing data negative reviews: ",len(test_data_neg))

    # Making out_sw.csv without labels for list with stopwords.
    out_sw=[]
    for i in train_data_pos:
        out_sw.append(i)
    for j in val_data_pos:
        out_sw.append(j)
    for k in test_data_pos:
        out_sw.append(k)
    for l in train_data_neg:
        out_sw.append(l)
    for m in val_data_neg:
        out_sw.append(m)
    for n in test_data_neg:
        out_sw.append(n)

    print('out_sw--------------')
    print(out_sw[0:10])

    with open('out_sw.csv','w') as f:
        for sublist in out_sw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+ ',') # adding string literal to make it easy to convert from csv to list of lists 
                #in the classification task in a2 for making ngrams. basically for making backtracking easy.
            f.write('\n')


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
    l_of_l_of_neg_rev_no_sw=List_without_stopwords[1]
    neg_index_list_no_sw=list(range(0,len(l_of_l_of_neg_rev_no_sw)))
    
    index_train_data_neg_no_sw=random.sample(neg_index_list_no_sw,k=int(0.8*(len(neg_index_list_no_sw))))
    remaining_indexes=list(set(neg_index_list_no_sw)-set(index_train_data_neg_no_sw))
    index_val_data_neg_no_sw=random.sample(remaining_indexes,int(0.5*(len(remaining_indexes))))
    remaining_indexes=list(set(remaining_indexes)-set(index_val_data_neg_no_sw))
    index_test_data_neg_no_sw=remaining_indexes
    
    for i in index_train_data_neg_no_sw:
        train_data_neg_no_sw.append(l_of_l_of_neg_rev_no_sw[i])
    for j in index_val_data_neg_no_sw:
        val_data_neg_no_sw.append(l_of_l_of_neg_rev_no_sw[j])
    for k in index_test_data_neg_no_sw:
        test_data_neg_no_sw.append(l_of_l_of_neg_rev_no_sw[k])
    
    print("Total instances negative reviews: ",len(neg_index_list_no_sw))
    print("Size of training data negative reviews: ",len(train_data_neg_no_sw))
    print("Size of validation data negative reviews: ",len(val_data_neg_no_sw))
    print("Size of testing data negative reviews: ",len(test_data_neg_no_sw))
    # We have 12 list of lists.


# Making out_nsw.csv without labels for list with no stopwords.
    out_nsw=[]
    for i in train_data_pos_no_sw: 
        out_nsw.append(i)
    for j in val_data_pos_no_sw:
        out_nsw.append(j)
    for k in test_data_pos_no_sw:
        out_nsw.append(k)
    for l in train_data_neg_no_sw:
        out_nsw.append(l)
    for m in val_data_neg_no_sw:
        out_nsw.append(m)
    for n in test_data_neg_no_sw:
        out_nsw.append(n)
    
    print('out_nsw--------------')
    print(out_nsw[0:10])


    with open('out_nsw.csv','w') as f:
        for sublist in out_nsw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+ ',')
            f.write('\n')

    # Adding labels to the files 
    train_data_pos1=copy.deepcopy(train_data_pos)
    val_data_pos1=copy.deepcopy(val_data_pos)
    test_data_pos1=copy.deepcopy(test_data_pos)
    train_data_neg1=copy.deepcopy(train_data_neg)
    val_data_neg1=copy.deepcopy(val_data_neg)
    test_data_neg1=copy.deepcopy(test_data_neg)
    train_data_pos_no_sw1=copy.deepcopy(train_data_pos_no_sw)
    val_data_pos_no_sw1=copy.deepcopy(val_data_pos_no_sw)
    test_data_pos_no_sw1=copy.deepcopy(test_data_pos_no_sw)
    train_data_neg_no_sw1=copy.deepcopy(train_data_neg_no_sw)
    val_data_neg_no_sw1=copy.deepcopy(val_data_neg_no_sw)
    test_data_neg_no_sw1=copy.deepcopy(test_data_neg_no_sw)
# Label insertion 
    # for i in train_data_pos1:
    #     i.insert(0,'1')

    # for i in val_data_pos1:
    #     i.insert(0,'1')

    # for i in test_data_pos1:
    #     i.insert(0,'1')

    # for i in train_data_neg1:
    #     i.insert(0,'0')

    # for i in val_data_neg1:
    #     i.insert(0,'0')

    # for i in test_data_neg1:
    #     i.insert(0,'0')

    # for i in train_data_pos_no_sw1:
    #     i.insert(0,'1')

    # for i in val_data_pos_no_sw1:
    #     i.insert(0,'1')

    # for i in test_data_pos_no_sw1:
    #     i.insert(0,'1')

    # for i in train_data_neg_no_sw1:
    #     i.insert(0,'0')

    # for i in val_data_neg_no_sw1:
    #     i.insert(0,'0')

    # for i in test_data_neg_no_sw1:
    #     i.insert(0,'0')
    # print("train sw with labels --_______--")
    # print(train_data_pos1[0:5])
    # print("neg")
    # print(train_data_neg1[0:5])
# concatinating train_pos,train_neg-->train and so on for all the files
# appending in base itself so train_data_pos1 will contain neg also for all.
    for i in train_data_neg1:
        train_data_pos1.append(i)
    train_sw=train_data_pos1

    for i in val_data_neg1:
        val_data_pos1.append(i)
    val_sw=val_data_pos1

    for i in test_data_neg1:
        test_data_pos1.append(i)
    test_sw=test_data_pos1

    for i in train_data_neg_no_sw1:
        train_data_pos_no_sw1.append(i)
    train_nsw=train_data_pos_no_sw1

    for i in val_data_neg_no_sw1:
        val_data_pos_no_sw1.append(i)
    val_nsw=val_data_pos_no_sw1

    for i in test_data_neg_no_sw1:
        test_data_pos_no_sw1.append(i)
    test_nsw=test_data_pos_no_sw1


    # print(test_nsw[38000:])
    # print(test_nsw[-1:-4])

    with open('train_sw.csv','w') as f:
        for sublist in train_sw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')
        
    with open('val_sw.csv','w') as f:
        for sublist in val_sw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')

    with open('test_sw.csv','w') as f:
        for sublist in test_sw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')

    with open('train_nsw.csv','w') as f:
        for sublist in train_nsw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')

    with open('val_nsw.csv','w') as f:
        for sublist in val_nsw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')

    with open('test_nsw.csv','w') as f:
        for sublist in test_nsw:
            for item in sublist:
                f.write(str(item)+',')
                # f.write('\''+str(item)+'\''+',')
            f.write('\n')
    # pos will have concatinated neg for all the l of l 
    # print("train data pos: ",len(train_data_pos1))
    # print("train data neg: ",len(train_data_neg1))
    # print("val data pos: ",len(val_data_pos1))
    # print("val data neg: ",len(val_data_neg1))
    # print("test data neg: ",len(val_data_neg1))