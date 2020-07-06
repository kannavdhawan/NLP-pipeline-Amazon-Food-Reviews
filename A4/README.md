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
relu | False | False | 78.94 | 74.23 | 74.31
relu | False | True(0.5) | 73.96 | 74.55 | 74.86    
relu | True(0.01) | True(0.5) | 63.20 | 70.74 | 70.55
relu | True(0.001) | True(0.5) | 65.60 | 69.91 | 69.90
tanh | False |False | 78.25 | 73.14 | 73.26
tanh | False |True(0.5) | 73.12 | 73.63 | 73.64    
tanh | True(0.01) | True(0.5) | 68.18 | 71.36 | 71.32
tanh | True(0.001) | True(0.5) | 70.37|  73.10 | 72.82
sigmoid | False | False | 83.84 | 74.35 | 74.18
sigmoid | False | True(0.5) | 74.84 | 74.99 | 74.45
sigmoid | True(0.01) | True(0.5) | 63.19 | 67.30 | 67.17
sigmoid | True(0.001) | True(0.5) | 68.93 | 72.02 | 71.92

Best Model at 0.2 Dropout:
Activation function | L2-norm regularization | Dropout | Train Accuracy(%) | Val Accuracy(%) | Test Accuracy(%)
--- | --- | --- | --- | --- | ---
relu | False | True(0.2) | 73.11 | 74.76 | 75.05   
tanh | False |True(0.2) | 73.25 | 73.97 | 74.44  
sigmoid | False |True(0.2) | 74.10 | 74.14 | 74.26    

# Analysis
__________________________________________
- Effect of activation functions on results (ReLU,tanh,sigmoid)
    - All the three activation functions provide almost similar results with a mean change in accuracy of (+-)1% for all the models.
    - ReLU works better than tanh and sigmoid by 1% of accuracy and it took 50s/epoch which is much less than the 120s(approx) taken by the other activation functions.
        ReLU being a ramp function, doesn't end up with vanishing gradient in case of deep networks whereas, sigmoid functions 
    - On the other and "tanh" works slightly better than the sigmoid in our case 
    If you use the hyperbolic tangent you might run into the fading gradient problem, meaning if x is 
    smaller than -2 or bigger than 2, the derivative gets really small and your network might not converge, or you might end up having a
     dead neuron that does not fire anymore.


- References:
    - https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
<!-- 


66% positive relu 90% neg  78
71% positive tanh 88    79.2
67% positive sig 92 79.5 -->
