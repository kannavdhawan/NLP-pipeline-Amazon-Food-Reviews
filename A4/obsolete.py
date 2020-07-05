# # import numpy as np
# # full_data=[['kannav','dhawan'],['kannav','dhawan','ajkakj'],['kannav','dhawan','ajkakj']]
# # print(int(np.percentile([len(seq) for seq in full_data], 95)))

# # print([' '.join(seq[:2]) for seq in full_data])
# # import numpy as np
# # import pandas as pd 
# # a=np.zeros((3,4))
# # b=pd.DataFrame(a)
# print(' '.join(['hello','kannav','dhawan']))

# full_data=[['I' ,'am', 'kannav', 'dhawan'],['I' ,'am', 'kannav', 'dhawan']]
# print([' '.join(seq[:2]) for seq in full_data])

import keras
import os 
import sys 
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from Neural_net import load_data,dataframe_to_l_of_l,fit_on_text
def main(path,model):
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
    print(data_p)
    #returning word index dict
    
    tup=load_data(data_path)
    l_of_l=dataframe_to_l_of_l(tup[-1])
    max_length,token=fit_on_text(l_of_l)


    #test data
    seq_test_data= token.texts_to_sequences(data_p)
    pad_test_data=pad_sequences(seq_test_data, maxlen=max_length, padding='post', truncating='post')
    final_test_data=pd.DataFrame(pad_test_data)
    #model load 

    model=keras.models.load_model(os.path.join('data/{}'.format(model)))
    print(model.summary())
    y_pred= model.predict_classes(final_test_data)
    print(y_pred)
    
if __name__=='__main__':
    main(os.sys.argv[1],os.sys.argv[2])



