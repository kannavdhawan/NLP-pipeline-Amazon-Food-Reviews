import pandas as pd
import numpy as np
import io
import random
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten,Conv1D
from keras import regularizers
random.seed(1332)

train=pd.read_csv('data/train_sw.csv',sep="\n",names=['Review'])
val=pd.read_csv('data/val_sw.csv',sep="\n",names=['Review'])
test=pd.read_csv('data/test_sw.csv',sep="\n",names=['Review'])
print("shape of train:",train.shape)
train_labels=pd.read_csv('data/train_sw_labels.csv',names=['label'])
val_labels=pd.read_csv('data/val_sw_labels.csv',names=['labels'])
test_labels=pd.read_csv('data/test_sw_labels.csv',names=['labels'])
train['label']=train_labels
val['label']=val_labels
test['label']=test_labels
train = train.sample(frac=1).reset_index(drop=True)
val = val.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)
X_train=train['Review']
y_train=train['label']
X_val=val['Review']
y_val=val['label']
X_test=test['Review']
y_test=test['label']
print(X_train.head())
data=pd.concat((X_train,X_val,X_test),axis=0)
def stringtolist(data):
    """
    Function taking dataframe as input.
    Returns: list of list. 

    """
    data=[(data.iloc[i])[:-1].split(',') for i in range(len(data))] # list in his case 
    print(type(data))
    print(data[0:4])
    return data
X_train=stringtolist(X_train) #lofl
X_val=stringtolist(X_val)#lofl
X_test=stringtolist(X_test)#lofl
data=stringtolist(data)#lofl
length = int(np.percentile([len(seq) for seq in data], 95)) # average number of words in each sentence.
print("Average sentence length:::::::::", length)

token=Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

"""
[' '.join(seq[:length]) for seq in data]=> 
"""

token.fit_on_texts([' '.join(seq[:length]) for seq in data])# input to fix_on_text: This product is very good.


X_train = token.texts_to_sequences([' '.join(seq[:length]) for seq in X_train])
X_val = token.texts_to_sequences([' '.join(seq[:length]) for seq in X_val])
X_test = token.texts_to_sequences([' '.join(seq[:length]) for seq in X_test])


print("X_train_after text to seq",X_train[0:5])
print("X_train_after text to seq",type(X_train))


X_train = pad_sequences(X_train, maxlen=length, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=length, padding='post', truncating='post')
# print("X_train_after pad seq",X_train)
print("X_train_after pad seq",type(X_train))

embeddings = Word2Vec.load('data/word2vec.model')
#vector size that I took 
EMB_DIM=embeddings.vector_size #350
print("vector size embedding",EMB_DIM)
#total number of words in t he dictionary.
VOCAB_SIZE=len(token.word_index)+1  #114556
print("VOCAB_SIZE",VOCAB_SIZE)

#to be done for test data 
embedding_matrix=np.random.randn(VOCAB_SIZE,EMB_DIM) #114556*350

for word,i in token.word_index.items():
  if word in embeddings.wv.vocab:
    embedding_matrix[i]=embeddings[word]
  else:
    embedding_matrix[i]=np.random.randn(1,EMB_DIM)

y_train=np_utils.to_categorical(y_train)
y_val=np_utils.to_categorical(y_val)
y_test=np_utils.to_categorical(y_test)

classifier=Sequential()
#114556 - vocab size . number of words in dict. word_index. each word 350 dim 
classifier.add(Embedding(input_dim=VOCAB_SIZE,output_dim=EMB_DIM,weights=[embedding_matrix], input_length=length,
                         trainable=False)) # Embedding layer

classifier.add(Flatten()) #flatten
classifier.add(Dense(64,activation='relu'))# hidden layer 
classifier.add(Dropout(0.3)) #dropout
classifier.add(Dense(2,activation='softmax',name='Output_Layer')) # final layer
classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(classifier.summary())

print("shape of train:",X_train.shape)
print("shape of val:",X_val.shape)
print("shape of test:",X_test.shape)
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)

X_val=pd.DataFrame(X_val)
y_val=pd.DataFrame(y_val)

print(X_train.head())

X_test=pd.DataFrame(X_test)
y_test=pd.DataFrame(y_test)

classifier.fit(X_train, y_train,
                  batch_size=1024,
                  epochs=15,
                  validation_data=(X_val, y_val))
print("Test Accuracy : " + str(classifier.evaluate(X_test,y_test)[1]*100))