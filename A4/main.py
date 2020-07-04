import numpy as np 
import pandas as pd 
import random
import os
import sys
from Neural_net import load_data,dataframe_to_l_of_l,fit_on_text,texts_to_sequences,embedding_matrix,to_df,model
def main(path):
    data=load_data(path) #returns (train,val,test,X_train,y_train,X_val,y_val,X_test,y_test,data)
    #unpacking
    X_train=data[3]
    y_train=data[4]
    X_val=data[5]
    y_val=data[6]
    X_test=data[7]
    y_test=data[8]
    data=data[9]

    X_train=dataframe_to_l_of_l(X_train) #dataframe to lofl
    X_val=dataframe_to_l_of_l(X_val)
    X_test=dataframe_to_l_of_l(X_test)
    data=dataframe_to_l_of_l(data)

    max_length,token=fit_on_text(data)
    X_train,X_val,X_test=texts_to_sequences(token,max_length,X_train,X_val,X_test) # return token?
    e_dim,v_size,embed_matrix=embedding_matrix(path,token)
    X_train,X_val,X_test,y_train,y_val,y_test=to_df(X_train,X_val,X_test,y_train,y_val,y_test)
    model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test)

if __name__=='__main__':
    main(os.sys.argv[1])