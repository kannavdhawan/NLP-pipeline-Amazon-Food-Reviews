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
    #|act_func,l2_norm_f,l2_norm,dropout_f,dropout |

    #Relu 
    # rel_acc_m1=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"relu",False,0,False,0) # dropout=False | l2 False
    # rel_acc_m2=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"relu",False,0,True,0.5)# dropout=True | l2 False
    #                                                 # No dropout leads to overfitting|dropout=0.5 | checking l2 rate 
    rel_acc_m3=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"relu",True,0.01,True,0.5) #dropout=True| l2 0.01
    rel_acc_m4=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"relu",True,0.001,True,0.5) #dropout=True| l2 0.001


#tanh 
    tanh_acc_m1=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"tanh",False,0,False,0) # dropout=False | l2 False
    tanh_acc_m2=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"tanh",False,0,True,0.5)# dropout=True | l2 False
                                                    # No dropout leads to overfitting|dropout=0.5 | checking l2 rate 
    tanh_acc_m3=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"tanh",True,0.01,True,0.5) #dropout=True| l2 0.01
    tanh_acc_m4=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"tanh",True,0.001,True,0.5) #dropout=True| l2 0.001

#sigmoid 
    sigmoid_acc_m1=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"sigmoid",False,0,False,0) # dropout=False | l2 False
    sigmoid_acc_m2=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"sigmoid",False,0,True,0.5)# dropout=True | l2 False
                                                    # No dropout leads to overfitting|dropout=0.5 | checking l2 rate 
    sigmoid_acc_m3=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"sigmoid",True,0.01,True,0.5) #dropout=True| l2 0.01
    sigmoid_acc_m4=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,"sigmoid",True,0.001,True,0.5) #dropout=True| l2 0.001


    print("\t\t\t\t~Activation: relu~")
    # print("L2=F | Dropout=F | Acc: ",rel_acc_m1)
    # print("L2=F | Dropout=T(0.5) | Acc: ",rel_acc_m2)
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",rel_acc_m3)
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",rel_acc_m4)

    print("\t\t\t\t~Activation: tanh~")
    print("L2=F | Dropout=F | Acc: ",tanh_acc_m1)
    print("L2=F | Dropout=T(0.5) | Acc: ",tanh_acc_m2)
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",tanh_acc_m3)
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",tanh_acc_m4)

    print("\t\t\t\t~Activation: sigmoid~")
    print("L2=F | Dropout=F | Acc: ",sigmoid_acc_m1)
    print("L2=F | Dropout=T(0.5) | Acc: ",sigmoid_acc_m2)
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",sigmoid_acc_m3)
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",sigmoid_acc_m4)

if __name__=='__main__':
    main(os.sys.argv[1])