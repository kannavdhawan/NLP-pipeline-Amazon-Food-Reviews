input:  [batchsize,textsize]-[256,24]
embedding: [batchsize,textsize,vecsize]-(256,24,350)
flatten (Flatten)->(None, 8400)  
Dense(None,120)
Dropout(None,120)
output_dense(None,2)




Relu, sigmoid,tanh
L2 norm regularization effect
dropouts effect







## code 1:

classifier=Sequential()
classifier.add(Embedding(input_dim=VOCAB_SIZE,output_dim=EMB_DIM,weights=[embedding_matrix], input_length=length,
                         trainable=False)) # Embedding layer

classifier.add(Flatten()) #flatten
classifier.add(Dense(120,activation='relu'))# hidden layer 
classifier.add(Dropout(0.5)) #dropout
classifier.add(Dense(2,activation='softmax',name='Output_Layer')) # final layer
classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(classifier.summary())

classifier.fit(X_train, y_train,
                  batch_size=128,
                  epochs=15,
                  validation_data=(X_val, y_val))

Epoch 15/15
5000/5000 [==============================] - 94s 19ms/step - loss: 0.5289 - accuracy: 0.7299 - val_loss: 0.5128 - val_accuracy: 0.7452
2500/2500 [==============================] - 10s 4ms/step - loss: 0.5133 - accuracy: 0.7445
Test Accuracy : 74.45250153541565



## code 2:

classifier=Sequential()
classifier.add(Embedding(input_dim=VOCAB_SIZE,output_dim=EMB_DIM,weights=[embedding_matrix], input_length=length,
                         trainable=False)) # Embedding layer

classifier.add(Flatten()) #flatten
classifier.add(Dense(120,activation='relu'))# hidden layer 
classifier.add(Dropout(0.5)) #dropout
classifier.add(Dense(2,activation='softmax',name='Output_Layer')) # final layer
classifier.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(classifier.summary())

classifier.fit(X_train, y_train,
                  batch_size=128,
                  epochs=15,
                  validation_data=(X_val, y_val))



Test Accuracy : 74.94750022888184


































































neurons: 64 

12th epoch |  loss: 0.4855 - accuracy: 0.7587 - val_loss: 0.5308 - val_accuracy: 0.7427
Test data | 2500/2500 [==============================] - 9s 4ms/step - loss: 0.5320 - accuracy: 0.7425
Test Accuracy : 74.24874901771545
gitpod /workspace/msci-text-analytics-s20 $ 


neurons: 128

Epoch 1/15
625/625 [==============================] - 111s 177ms/step - loss: 0.6481 - accuracy: 0.6346 - val_loss: 0.5521 - val_accuracy: 0.7096
Epoch 2/15
625/625 [==============================] - 126s 202ms/step - loss: 0.5533 - accuracy: 0.7116 - val_loss: 0.5254 - val_accuracy: 0.7323
Epoch 3/15
625/625 [==============================] - 98s 157ms/step - loss: 0.5392 - accuracy: 0.7241 - val_loss: 0.5222 - val_accuracy: 0.7367
Epoch 4/15
625/625 [==============================] - 86s 138ms/step - loss: 0.5288 - accuracy: 0.7317 - val_loss: 0.5158 - val_accuracy: 0.7416
Epoch 5/15
625/625 [==============================] - 81s 130ms/step - loss: 0.5211 - accuracy: 0.7388 - val_loss: 0.5132 - val_accuracy: 0.7433
Epoch 6/15
625/625 [==============================] - 81s 129ms/step - loss: 0.5143 - accuracy: 0.7431 - val_loss: 0.5115 - val_accuracy: 0.7473
Epoch 7/15
625/625 [==============================] - 81s 130ms/step - loss: 0.5101 - accuracy: 0.7455 - val_loss: 0.5156 - val_accuracy: 0.7441
Epoch 8/15
625/625 [==============================] - 79s 127ms/step - loss: 0.5042 - accuracy: 0.7491 - val_loss: 0.5117 - val_accuracy: 0.7470
Epoch 9/15
625/625 [==============================] - 83s 132ms/step - loss: 0.4999 - accuracy: 0.7511 - val_loss: 0.5148 - val_accuracy: 0.7484
Epoch 10/15
625/625 [==============================] - 86s 138ms/step - loss: 0.4959 - accuracy: 0.7532 - val_loss: 0.5187 - val_accuracy: 0.7475
Epoch 11/15
625/625 [==============================] - 81s 129ms/step - loss: 0.4921 - accuracy: 0.7552 - val_loss: 0.5164 - val_accuracy: 0.7507
Epoch 12/15
625/625 [==============================] - 80s 128ms/step - loss: 0.4888 - accuracy: 0.7575 - val_loss: 0.5199 - val_accuracy: 0.7486
Epoch 13/15
625/625 [==============================] - 79s 127ms/step - loss: 0.4862 - accuracy: 0.7593 - val_loss: 0.5258 - val_accuracy: 0.7461
Epoch 14/15
625/625 [==============================] - 79s 126ms/step - loss: 0.4837 - accuracy: 0.7600 - val_loss: 0.5200 - val_accuracy: 0.7531
Epoch 15/15
625/625 [==============================] - 81s 130ms/step - loss: 0.4803 - accuracy: 0.7613 - val_loss: 0.5214 - val_accuracy: 0.7519
2500/2500 [==============================] - 9s 4ms/step - loss: 0.5199 - accuracy: 0.7514
Test Accuracy : 75.14374852180481







1. relu | tanh | sigmoid 
2. l2 change 
3. dropouts change 

1. relu l2 change dropout fix => l2 fix 
2. tanh l2 change dropout
3. sigmoid 

|Activation function|L2-norm regularization|Dropouts|

|relu|No|0.5|
|relu||0.01|0.5|
|relu|0.001|0.5|
|relu|best|0.3|
best accuracy =>

|tanh|No|0.5|
|tanh||0.01|0.5|
|tanh|0.001|0.5|
|tanh|best|0.3|
best accuracy =>

|sigmoid|No|0.5|
|sigmoid||0.01|0.5|
|sigmoid|0.001|0.5|
|sigmoid|best|0.3|
best accuracy =>


Citations: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html