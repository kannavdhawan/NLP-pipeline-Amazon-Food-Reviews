import pandas as pd
import numpy as np
import os
import random
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

