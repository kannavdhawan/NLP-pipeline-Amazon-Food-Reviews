import numpy as np 
import pandas as pd 
import random
import os
import sys
from Neural_net import load_data,dataframe_to_l_of_l,fit_on_text,texts_to_sequences,embedding_matrix,to_df,model
import matplotlib.pyplot as plt
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

    a_f_list=["relu","tanh","sigmoid"]
    
    acc_list=[]
    hist_list=[]
    for act_f in a_f_list:
        print("Model ",act_f)
        acc1,history1=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,act_f,False,0,False,0) # dropout=False | l2 False
        acc2,history2=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,act_f,False,0,True,0.2)# dropout=True | l2 False
                                                            # No dropout leads to overfitting|dropout=0.5 | checking l2 rate 
        acc3,history3=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,act_f,True,0.01,True,0.5) #dropout=True| l2 0.01
        acc4,history4=model(X_train,X_val,X_test,max_length,e_dim,v_size,embed_matrix,y_train,y_val,y_test,act_f,True,0.0001,True,0.5) #dropout=True| l2 0.001
        
        acc_list.extend([acc1,acc2,acc3,acc4])
        hist_list.extend([history1,history2,history3,history4])
        
  

    """
    plot for loss of different act functions. 
    """
    plt.plot(hist_list[1].history['val_loss'])
    plt.plot(hist_list[5].history['val_loss'])
    plt.plot(hist_list[9].history['val_loss'])
    plt.title('Model Loss with different parameters')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['ReLU', 'tanh','sigmoid'], loc='upper left')
    plt.show()
    plt.savefig('loss.png')




    print("\t\t\t\t~Activation: relu~")
    print("L2=F | Dropout=F | Acc: ",acc_list[0])
    print("L2=F | Dropout=T(0.5) | Acc: ",acc_list[1])
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",acc_list[2])
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",acc_list[3])

    print("\t\t\t\t~Activation: tanh~")
    print("L2=F | Dropout=F | Acc: ",acc_list[4])
    print("L2=F | Dropout=T(0.5) | Acc: ",acc_list[5])
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",acc_list[6])
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",acc_list[7])

    print("\t\t\t\t~Activation: sigmoid~")
    print("L2=F | Dropout=F | Acc: ",acc_list[8])
    print("L2=F | Dropout=T(0.5) | Acc: ",acc_list[9])
    print("L2=T(0.01) | Dropout=T(0.5) | Acc: ",acc_list[10])
    print("L2=T(0.001) | Dropout=T(0.5) | Acc: ",acc_list[11])

if __name__=='__main__':
    main(os.sys.argv[1])