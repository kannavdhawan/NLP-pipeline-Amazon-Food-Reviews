import keras
import os 
import sys 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json
from keras.preprocessing.text import tokenizer_from_json
from Neural_net import load_data,dataframe_to_l_of_l,fit_on_text
import numpy as np
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(path,model_code):
    data_path='data/'
    with open(path,'r') as f:
        data=f.read().splitlines()
    # print(data)
    #removing '.'
    data_p=[]
    for i in range(len(data)):
        if data[i][-1]=='.':
            tmp=data[i].replace('.','')
        else:
            tmp=data[i]
        data_p.append(tmp)
    # print(data_p)
    
    """returning word index dict from functions defined in Neural_net. Please uncomment if not using .json dict.
    """

    # tup=load_data(data_path)
    # l_of_l=dataframe_to_l_of_l(tup[-1])
    # max_length,token=fit_on_text(l_of_l)

    #loading saved dict 
    with open(os.path.join(data_path,'tokenizer.json')) as f:
        data_json = json.load(f)
    token= tokenizer_from_json(data_json)
    max_length=25

    #test data
    seq_test_data= token.texts_to_sequences(data_p)
    pad_test_data=pad_sequences(seq_test_data, maxlen=max_length, padding='post', truncating='post')
    final_test_data=pd.DataFrame(pad_test_data)
    
    #model load 
    if model_code=="relu":
        model=keras.models.load_model(os.path.join('data/{}'.format("nn_relu.model")))
    elif model_code=="tanh":
        model=keras.models.load_model(os.path.join('data/{}'.format("nn_tanh.model")))
    elif model_code=="sigmoid":
        model=keras.models.load_model(os.path.join('data/{}'.format("nn_sigmoid.model")))
    
    # print(model.summary())
    y_pred=model.predict(final_test_data)
    y_pred=np.argmax(y_pred, axis=-1)
    y_pred=y_pred.tolist()
    print("Final labels: ",y_pred)
    
    y_pred=['Positive' if x==1 else 'Negative' for x in y_pred]
    for i in range(len(data)):
        print(data[i][0:25],".. ===>",y_pred[i])
    
    
if __name__=='__main__':
    main(os.sys.argv[1],os.sys.argv[2])