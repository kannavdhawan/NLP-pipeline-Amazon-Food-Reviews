import pandas as pd
import numpy as np
import io
import random
import os
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten,Conv1D
from keras import regularizers
random.seed(1332)
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    """
    Input: path--> data/
    Returns: train,val,test,X_train,y_train,X_val,y_val,X_test,y_test,data
    """
    
    train=pd.read_csv(os.path.join(path,'train_sw.csv'),sep="\n",names=['Review'])
    val=pd.read_csv(os.path.join(path,'val_sw.csv'),sep="\n",names=['Review'])
    test=pd.read_csv(os.path.join(path,'test_sw.csv'),sep="\n",names=['Review'])

    print("\t\t\t\tTrain head\n",train.head())
    print("shape of train:",train.shape)
    print("shape of val:",val.shape)
    print("shape of test:",test.shape)

    train_labels=pd.read_csv(os.path.join(path,'train_sw_labels.csv'),names=['label'])
    val_labels=pd.read_csv(os.path.join(path,'val_sw_labels.csv'),names=['labels'])
    test_labels=pd.read_csv(os.path.join(path,'test_sw_labels.csv'),names=['labels'])

    # print("\t\t\t\tTrain labels head: \n",train_labels.head())
    # print("\t\t\t\tTrain labels tail: \n",train_labels.tail())

    #Adding corresponding 
    train['label']=train_labels
    val['label']=val_labels
    test['label']=test_labels

    #shuffling 
    """
    Model will reach a better minima. 
    """

    train = train.sample(frac=1).reset_index(drop=True)
    print("\t\t\t\tTrain head:\n",train.head())
    val = val.sample(frac=1).reset_index(drop=True)
    print("\t\t\t\tVal head:\n",val.head())
    test = test.sample(frac=1).reset_index(drop=True)
    print("\t\t\t\tTest head:\n",test.head())

    """
    Implicit labelling If required. Uncomment below till line 73 and comment above from line 24-33
    """

    # L1_train=pd.Series([int(1)]*int((len(train)/2)))
    # L0_train=pd.Series([int(0)]*int((len(train)/2)))
    # train_labels=pd.concat([L1_train,L0_train],ignore_index=True)
    # train['label']=train_labels

    # L1_val=pd.Series([int(1)]*int((len(val)/2)))
    # L0_val=pd.Series([int(0)]*int((len(val)/2)))
    # val_labels=pd.concat([L1_val,L0_val],ignore_index=True)
    # val['label']=val_labels

    # L1_test=pd.Series([int(1)]*int((len(test)/2)))
    # L0_test=pd.Series([int(0)]*int((len(test)/2)))
    # test_labels=pd.concat([L1_test,L0_test],ignore_index=True)
    # test['label']=test_labels

    X_train=train['Review']
    y_train=train['label']

    X_val=val['Review']
    y_val=val['label']

    X_test=test['Review']
    y_test=test['label']

    # print(X_train.head())

    #concatinating all data without labels for findingthe max length and creating the word index 
    data=pd.concat((X_train,X_val,X_test),axis=0)
    
    # Converting into categorical data 
     
    y_train=np_utils.to_categorical(y_train)
    y_val=np_utils.to_categorical(y_val)
    y_test=np_utils.to_categorical(y_test)


    return train,val,test,X_train,y_train,X_val,y_val,X_test,y_test,data

def dataframe_to_l_of_l(data):
    """
    Function taking dataframe as input.
    Returns: list of list.
    """
    data=[(data.iloc[i])[:-1].split(',') for i in range(len(data))]
    # print(type(data))
    # print(data[0:4])
    return data


def fit_on_text(data):
    
    """
    =>Setting max length for pad_sequences and embedding layer.
    =>Embedding layer is capable of processing sequence of heterogenous length, if you don't pass an explicit input_length 
        argument to the layer).
    =>85% returns 21 . 95% returned 24. 98% provides 25 .  | Acc : 24>21>25 => selecting 95th %
    """
    max_length= int(np.percentile([len(seq) for seq in data],95)) # average number of words in each sentence.    
    
    """
    =>Updates internal vocabulary based on a list of texts.
    """
    
    token=Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    """
    =>fit_on_texts: Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency.
    =>So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word ->
    =>index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word (often the first few
      are stop words because they appear a lot).

    =>full_data=[['I' ,'am', 'kannav', 'dhawan'],['I' ,'am', 'kannav', 'dhawan']]
    =>print([' '.join(seq[:2]) for seq in full_data])
    =>op of join:
                ['I am', 'I am']>> Input to fit_on_texts

    =>Returns: A dictionary word_index with values as indexes . lowest one is most frequent.
    """
    
    list_of_strings_full_data=[' '.join(seq[:]) for seq in data]
    token.fit_on_texts(list_of_strings_full_data)# input to fix_on_text: ['This product is very good','']
    
    return max_length,token

def texts_to_sequences(token,max_length,X_train,X_val,X_test):
    """
    =>Texts_to_sequences Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces it with its 
    =>Corresponding integer value from the word_index dictionary.
    =>Only words known by the tokenizer will be taken into account.
    =>List of list where eah elment is a number/int taken from word_index dict.

    =>Why don't combine them? Because you almost always fit once and convert to sequences many times. You will fit on your training corpus once and use 
    that exact same word_index dictionary at train / eval / testing / prediction time to convert actual text into sequences to feed them to the network. 
    So keeping those methods separate.
    """
    tr=[' '.join(seq[:]) for seq in X_train]  #['This product is very good','']
    vl=[' '.join(seq[:]) for seq in X_val]    #['This product is very good','']
    tst=[' '.join(seq[:]) for seq in X_test]  #['This product is very good','']
    X_train = token.texts_to_sequences(tr)
    X_val = token.texts_to_sequences(vl)
    X_test = token.texts_to_sequences(tst)

    # print("X_train_after text to seq",X_train)
    print("X_train_after text to seq",type(X_train))
    """
    This function transforms a list (of length num_samples) of sequences (lists of integers) 
    into a 2D Numpy array of shape (num_samples, num_timesteps). 
    num_timesteps is either the maxlen argument if provided, or the length of the
    longest sequence in the list.
    """
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
    X_val = pad_sequences(X_val, maxlen=max_length, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
    print("shape test 1:",X_train.shape)
    print("shape test 2:",X_val.shape)
    print("shape test 3:",X_test.shape)

    # print("X_train_after pad seq",X_train)
    print("X_train_after pad seq",type(X_train))
    return X_train,X_val,X_test

def embedding_matrix(path,token):
    """
    Loading the embeddings created by w2v for feeding the corresponding vectors in embedding matrix.
    """
    w2v_embeddings = Word2Vec.load(os.path.join(path,'word2vec.model'))
    #vector size that I took
    e_dim=w2v_embeddings.vector_size #350
    print("vector size embedding",e_dim)
    #total number of words in the dictionary.
    v_size=len(token.word_index)+1  #114556 vocab size
    print("vocabulary_size: ",v_size)
    
    # making the embedding matrtix and feeding with array from word2vec embeddings.

    embed_matrix=np.random.randn(v_size,e_dim) #114556*350
    for word,index in token.word_index.items():
        if word in w2v_embeddings.wv.vocab:
            embed_matrix[index]=w2v_embeddings[word]#feeding the embedding matrix with array from word2vec embeddings
        else:
            embed_matrix[index]=np.random.randn(1,e_dim) # if word from word index is not there in word2vec embeddings, input randomly.
    


    return e_dim,v_size,embed_matrix

def to_df(X_train,X_val,X_test,y_train,y_val,y_test):
    
    print("shape of train:",X_train.shape)
    print("shape of val:",X_val.shape)
    print("shape of test:",X_test.shape)

    X_train=pd.DataFrame(X_train)
    y_train=pd.DataFrame(y_train)
    X_val=pd.DataFrame(X_val)
    y_val=pd.DataFrame(y_val)
    X_test=pd.DataFrame(X_test)
    y_test=pd.DataFrame(y_test)
    return X_train,X_val,X_test,y_train,y_val,y_test

def model(X_train,X_val,X_test,max_length,e_dim,v_size,e_mat,y_train,y_val,y_test):
    clf=Sequential()
    """114556 - vocab size . number of words in dict. word_index.| each word 350 dim 
    All that the Embedding layer does is to map the integer inputs to the vectors found at the corresponding index in the embedding matrix,
    i.e. the sequence [1, 2] would be converted to [embeddings[1], embeddings[2]] i.e. the correspondimg 350 sized vector
    .This means that the output of the Embedding layer will be a 3D tensor of shape (samples, sequence_length, embedding_dim).
    """
    e_layer=Embedding(input_dim=v_size,output_dim=e_dim,weights=[e_mat], input_length=max_length,
                            trainable=False)
    clf.add(e_layer) # Embedding layer

    #(114556,350,[114556*350],24)==>(None,24,350) i.e. (input_length,output_dim)
    clf.add(Flatten()) #flatten
    clf.add(Dense(128,activation='relu'))# hidden layer 
    clf.add(Dropout(0.3)) #dropout
    clf.add(Dense(2,activation='softmax'))# final layer
    clf.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    print(clf.summary())


    clf.fit(X_train, y_train,batch_size=1024,epochs=2,validation_data=(X_val, y_val))
    
    test_acc=clf.evaluate(X_test,y_test)[1]
    print("Test Accuracy : ", test_acc*100)
    clf.save('data/relu_test.model')