# import pandas as pd
# # from gensim.models import Word2Vec
# train_sw=pd.read_csv("csv_splits/train_sw.csv",sep=';',header=None,names=['Reviews'])
# # print(train_sw.head(3))
# # train_nsw=pd.read_csv("train_nsw.csv",sep=';',header=None,names=['Reviews'])
# print(train_sw.iloc[0:5,:])
# # val_sw=pd.read_csv("val_sw.csv",sep=';',header=None,names=['Reviews'])
# # val_nsw=pd.read_csv("val_nsw.csv",sep=';',header=None,names=['Reviews'])
# # test_sw=pd.read_csv("test_sw.csv",sep=';',header=None,names=['Reviews'])
# # test_nsw=pd.read_csv("test_nsw.csv",sep=';',header=None,names=['Reviews'])
# # dataset=pd.concat([train_sw,train_nsw,val_sw,val_nsw,test_sw,test_nsw],axis=0,ignore_index=True)
# # # print(dataset.info())
# # # print(len(dataset))
# formatted_dataset=[]
# for i in range(len(dataset)):
#     temp=(dataset.iloc[i,0])[:-1].split(',')
#     formatted_dataset.append(temp)
# print(formatted_dataset[0:5])
# # #Model

# #creating the word vectors
# model=Word2Vec(sentences=formatted_dataset,min_count=15,window=2, size=300, sample=6e-5, alpha=0.03,min_alpha=0.0007, negative=20)
# model.save("word2vec.model")

# model=Word2Vec.load("word2vec.model")
# #“wv” stands for “word vectors”. used to acces the word vectors from the model.

# vector1=model.wv['good'] #a vector with n dimensions(300) for us. 
# # print(vector1)

# word_vectors=model.wv

# print(model.wv.vocab) # A dictionary with {word: keyvectors.Vocab object}
# # print(type(model.wv)) # word to keyed vectors class.

# # find
# # 1. model. build vocab,
# # @ 2. train 

# model.build_vocab(formatted_dataset,progress_per=10000)
# model.train(formatted_dataset, total_examples=model.corpus_count, epochs=15, report_delay=1)

# #Function to print most similar words to any given word within the vocabulary
# def similarWordTo(word, n):
#   for i in range(n):
#     print(model.wv.most_similar(positive=[word], topn=n)[i][0])

# print("Good: Most similar words:")
# similarWordTo("good",20)
# print("Bad: Most similar words:")
# similarWordTo("bad",20)
# import os

# with open(os.path.join('data/','oye.txt'),'w') as f:
#     pos='{}'.format("")*1000000000
#     f.write(pos)
# f.close

    