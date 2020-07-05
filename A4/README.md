<!-- input:  [batchsize,textsize]-[256,24]
embedding: [batchsize,textsize,vecsize]-(256,24,350)
flatten (Flatten)->(None, 8400)  
Dense(None,120)
Dropout(None,120)
output_dense(None,2) -->


- Width of the network at H.L
    - neurons: 64 
        - accuracy: 0.7587 | val_accuracy: 0.7427
        - Test Accuracy: 74.24%
    - neurons: 128
        - accuracy: 0.7613 | val_accuracy: 0.7519
        - Test Accuracy: 75.14%
    - Note: Selecting 128 neurons in H.L

Activation function | L2-norm regularization | Dropout | Train Accuracy(%) | Val Accuracy(%) | Test Accuracy(%)
--- | --- | --- | --- | --- | ---
relu | False | False | 77.61 | 73.86 | 74.14
relu | False | True(0.5) | 74.31 | 74.34 | 74.39    
relu | True(0.01) | True(0.5) | x | x | x
relu | True(0.001) | True(0.5) | x | x | x
tanh | False |False | x | x | x
tanh | False |True(0.5) | x | x | x |   
tanh | True(0.01) | True(0.5) | x | x | x
tanh | True(0.001) | True(0.5) | x|  x | x
sigmoid | False | False | x | x | x
sigmoid | False | True(0.5) | x | x | x
sigmoid | True(0.01) | True(0.5) | x | x | x
sigmoid | True(0.001) | True(0.5) | x | x | x

- References:
    - https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html