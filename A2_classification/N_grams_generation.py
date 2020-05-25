import pandas as pd
import numpy as np
import nltk
from sklearn.naive_bayes import MultinomialNB

# Labeling the TRAIN dataset
train_sw=pd.read_csv("train_sw.csv",sep=';',header=None)

L1_train=pd.Series([int(1)]*320000)
L0_train=pd.Series([int(0)]*320000)
train_labels=pd.concat([L1_train,L0_train],ignore_index=True)
train_sw['label']=train_labels
train_sw.columns = ['Text', 'Labels']
# print(train_labels.iloc[319999:320002])
# Labeling the TRAIN NSW dataset
train_nsw=pd.read_csv("train_nsw.csv",sep=';',header=None)
train_nsw['label']=train_labels
train_nsw.columns = ['Text', 'Labels']
print(train_sw.info())


# Labeling the VAL dataset
val_sw=pd.read_csv("val_sw.csv",sep=';',header=None)

L1_val=pd.Series([int(1)]*40000)
L0_val=pd.Series([int(0)]*40000)
val_labels=pd.concat([L1_val,L0_val],ignore_index=True)
val_sw['label']=val_labels
val_sw.columns = ['Text', 'Labels']
# Labeling the VAL NSW dataset
val_nsw=pd.read_csv("val_nsw.csv",sep=';',header=None)
val_nsw['label']=val_labels
val_nsw.columns = ['Text', 'Labels']
print(val_sw.info())


# Labeling the TEST dataset
test_sw=pd.read_csv("test_sw.csv",sep=';',header=None)

L1_test=pd.Series([int(1)]*40000)
L0_test=pd.Series([int(0)]*40000)
test_labels=pd.concat([L1_test,L0_test],ignore_index=True)
test_sw['label']=test_labels
test_sw.columns = ['Text', 'Labels']
# Labeling the TEST NSW dataset
test_nsw=pd.read_csv("test_nsw.csv",sep=';',header=None)
test_nsw['label']=test_labels
test_nsw.columns = ['Text', 'Labels']
print(test_sw.info())

print("train sw :")
print(train_sw.head(3))

# Converting into list of list without labels 
#-------------FOLLOWING CODE CAN BE USED IF CSV CONTAINS STRINGS IN THE FORM OF '' .
# will not work because of words like didn't leave t outside the two strings
# for i in range(len(train_sw['Text'])):
#     train_sw_list.append(list(eval(train_sw.iloc[i,0])))
# print(train_sw_list[0:5])

# train_sw_list.append(eval('['+train_sw.iloc[20,0]+']'))
# print(train_sw_list)

# Extarcting training data into list of list 
temp_train_sw=[]
train_sw_list=[]
for i in range(len(train_sw)):
    temp_train_sw=(train_sw.iloc[i,0])[:-1].split(',')
    train_sw_list.append(temp_train_sw)
    # print((train_sw.iloc[i,0])[:-1]) # slicing and removing extra comma which can be simply removed using eval otherwise. 
    # print(temp_train_sw)
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
# uncomment below_________________________-----------------------------------______________________________-
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

#unigrams
u=1                                                                # stopwords set-1 
unigram_train_sw=n_grams(train_sw_list,u)
unigram_val_sw=n_grams(val_sw_list,u)
unigram_test_sw=n_grams(test_sw_list,u)
print("set 1: test ",len(unigram_test_sw))
u=1                                                                # No stopwords set-2
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

u=2                                                                # No stopwords set-4
bigrams_train_nsw=n_grams(train_nsw_list,u)
bigrams_val_nsw=n_grams(val_nsw_list,u)
bigrams_test_nsw=n_grams(test_nsw_list,u)
print("set 4: train ",len(bigrams_train_nsw))

# unigram+bigrams 

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

# No stopwords set-6
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


def dic(j):
    return dict([(word, True) for word in j]) # returns dictionary
def data_formatting(input,label_size):
    for i in range(len(input)):
        if i<=label_size:
            temp=[(dic(j),1) for j in input] #format-->  [({},1)]
        else:
            temp=[(dic(j),0) for j in input]
    return temp
print("1")
unigram_train_sw_final=data_formatting(unigram_train_sw,320000)
print(unigram_train_sw_final[319999])
print(unigram_train_sw_final[320000])
print("2")
unigram_val_sw_final=data_formatting(unigram_val_sw,40000)
print("3")
unigram_test_sw_final=data_formatting(unigram_test_sw,40000)

print("4")
unigram_train_nsw_final=data_formatting(unigram_train_nsw,320000)
print("5")
unigram_val_nsw_final=data_formatting(unigram_val_nsw,40000)
print("6")
unigram_test_nsw_final=data_formatting(unigram_test_nsw,40000)

print("7")
bigrams_train_sw_final=data_formatting(bigrams_train_sw,320000)
print("8")
bigrams_val_sw_final=data_formatting(bigrams_val_sw,40000)
print("9")
bigrams_test_sw_final=data_formatting(bigrams_test_sw,40000)

bigrams_train_nsw_final=data_formatting(bigrams_train_nsw,320000)
bigrams_val_nsw_final=data_formatting(bigrams_val_nsw,40000)
bigrams_test_nsw_final=data_formatting(bigrams_test_nsw,40000)


ub_train_sw_final=data_formatting(ub_train_sw,320000)
ub_val_sw_final=data_formatting(ub_val_sw,40000)
ub_test_sw_final=data_formatting(ub_test_sw,40000)

ub_train_nsw_final=data_formatting(ub_train_nsw,320000)
ub_val_nsw_final=data_formatting(ub_val_nsw,40000)
ub_test_nsw_final=data_formatting(ub_test_nsw,40000)
