
# Text analysis using corpus of amazon reviews.
__________

______________
______________
______________
______________
# Preprocessing
-------------------------------------------------

# Instructions to run the code:
From the root of this repository, run `python3 A1_Preprocessing/main.py data/` .

____________________________________
# main.py: 
    Data is loaded and all the modules are called over here.
    Functions in Tokenizer, Filter_sc, Stopwords_split are called.
____________________________________
# Tokenizer.py:
    Input: pos, neg text files
    Sentence tokenization. 
    Word tokenization. 

Returns a list of tokens consisting of [tokenized_pos,tokenized_neg].
____________________________________
# Filter_sc.py:
    Input: special_char_list, ob_tokenizer which consists of [tokenized_pos,tokenized_neg] generated in previous step.
    A function "spaced_special_char_filter" which removes separated special characters which are not joined with any other word for both the positive and negative tokens.
    A function "spec_char_filter" accepting special character list and ob_parially_filtered which is the list of partially filtered tokens from the previous step and this removes the special characters which are concatinated with other characters For instance: '@#method%^' => outputs 'method'.

Returns a list final_tokens consisting of filtered positive and negative tokens without special characters. 
____________________________________

# Stopwords_split.py
    Input: stopwords_list,List_with_stopwords.
    stopwords_remover removes stopwords from both the positive and negative reviews. 
    train_val_test splits the lists [List_with_stopwords,List_without_stopwords] into
        out_sw<=stopwords list=>train_pos,val_pos,test_pos.............train_neg,val_neg,test_neg
        out_nsw<=No stopwords list=>train_pos,val_pos,test_pos.............train_neg,val_neg,test_neg
        train_sw<=train_pos+train_neg
        val_sw<=val_pos+val_neg
        test_sw<=test_pos+test_neg
        train_nsw<=train_pos_nsw+train_neg_nsw
        val_nsw<=val_pos_nsw+val_neg_nsw
        test_nsw<=test_pos_nsw+test_neg_nsw
    Data distribution: 
        80% pos and 80% neg constitute the train data, total 640000 datapoints
        10% pos and 10% neg constitute the val data, total 80000 datapoints
        10% pos and 10% neg constitute the test data, total 80000 datapoints
    Note:
        1. Reviews are Randomly sampled by indexes while index generation.
        2. Not adding [] and '' here while creating csv's. code is commented. It can be created like that.  
        3. I have created simple features of words. In classification/Assignment 2, each line/review in csv is handeled as ['word1','word2'....]
        4. In order to create no bias, I have created the equal partitions of pos,neg reviews for train,val and test for both with and without stopwords. 

______________
______________
______________
______________



### Instructions to Run: 
`python3 A2_classification/main.py data/`
##### Learn submission is same as mentioned in additional guidelines.
# Results:
### Test set Accuracy:
| stopwords Removed  | Text features | Accuracy(Test set)|
| ------------------ | ------------- |-------------------|
|        Yes         |    unigrams   |       80.64%      | 
|        Yes         |    bigrams    |       78.32%      | 
|        Yes         |    uni+bi     |       82.43%      | 
|        No          |    unigrams   |       80.52%      | 
|        No          |    bigrams    |       81.75%      | 
|        No          |    uni+bi     |       82.71%      |

__________________________________________________________
### Validation set Accuracy:
- Hyperparameter tuning with best resulted hyperparameter(alpha), alpha: additive smoothing to supress the effect of rare words. 
    - For instance:
        - If there is only 1 neg review out of 20 reviews in the training set, then without having significant additive smoothing, model will classify
        the test data as negative if that word contributing to negative review is there in any of the review.

| stopwords Removed  | Text features | Accuracy(Val set) |  alpha(Best) |
| ------------------ | ------------- |-------------------|--------------|
|        Yes         |    unigrams   |       80.87%      |      0.5     |
|        Yes         |    bigrams    |       78.49%      |      1.0     |
|        Yes         |    uni+bi     |       82.64%      |      1.0     |
|        No          |    unigrams   |       80.71%      |      0.5     |
|        No          |    bigrams    |       82.03%      |      0.4     |
|        No          |    uni+bi     |       83.00%      |      0.4     |
________________________________________________________________________

# Analysis: 
### Performance comparison:  with and without stopwords:  
- Overall in my case, Models trained with stopwords performed better than models without stopwords by a small difference 
in Accuracy of +0.30 in {uni+bi} and a difference of +3.43% in {Bigrams} with an exception of {Unigrams} where the model without 
stopwords outperforms the one with stopwords by a negligible accuracy of +0.12%.
- Ideally, If the valence/context is not getting affected by removing SW, Accuracy(NO_SW)>Accuracy(SW).
- Performance is dependent on dataset and the list of stopwords which are removed. For instance, 
    - When SW "Don't" is removed, 2596 words with "Don't" contributing to sentiment in test_sw.csv are removed in test_nsw.csv.
    - Some Review Example set:
        - my,iphone,does,feel,protected,and,i,don't,have,to,worry,as,much,about,dropping,my,new,phone.
        - i,also,like,the,fact,that,i,don't,have,to,worry,about,maintaining,a,fancy,handle.
        - i,don't,see,it,as,a,big,problem,unless,for,some,reason,you,are,in,a,huge,rush.
    - These reviews on removal of SW "Don't" and other SW's contributing to the sentiment can be misclassified.
    - Thus, our accuracy with stopwords> without stopwords because valence/context is getting affected after removing SW.
- Particularly, in bigrams the accuracy difference(SW and n_SW) is huge(3.43%) because the same SW's words are contributing twice at each "nth" token in a review as compared to
 unigrams.
### Performance comparison:  unigrams, bigrams, unigrams+bigrams:
- Accuracy(unigrams+bigrams)>Accuracy(bigrams)>Accuracy(unigrams) | with stopwords.
    - {unigrams + bigrams} are preserving more information than the others and making the feature vector larger in size as well, so with an increase in the accuracy, we are compromising with the space and time complexity.
- Accuracy(unigrams+bigrams)>Accuracy(unigrams)>Accuracy(bigrams) | without stopwords. 
    - Reason being the same as above comparison, absence of useful words in multiple tokens leading to less accuracy in this set for bigrams.
    - As long as we are increasing the N_grams, accuracy is decreasing(If previous ((N-1) to 1) grams are not included).
- Overall, 
    - If only non essential stopwords are removed(based on the dataset), Accuracy(unigrams+bigrams)>Accuracy(bigrams)>Accuracy(unigrams) for both the sets.

##### Note: For method followed, please refer commented parts in Readme and .py files.Thanks.
### Citations for libraries used in code:
1. https://www.nltk.org/_modules/nltk/classify/scikitlearn.html 
2. https://scikit-learn.org/stable/modules/feature_extraction.html
3. https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52
4. https://stackoverflow.com/questions/25155940/nltk-naivebayesclassifier-input-formatting
5. http://www.sfs.uni-tuebingen.de/~keberle/NLPTools/presentations/NLTK/NLTK_Classifiers.pdf




<!-- output -->



<!-- 
---------------------------------unigram stopwords----------------------------------------------
Unigram sw Val acc at alpha= 0.1  is  0.8055875
Unigram sw Val acc at alpha= 0.4  is  0.8069625
Unigram sw Val acc at alpha= 0.5  is  0.80715
Unigram sw Val acc at alpha= 1.0  is  0.8067875
Unigram sw Val acc at alpha= 1.5  is  0.8062875


Unigrams sw val Best accuracy= 0.80715  at alpha= 0.5
Unigrams sw test accuracy= 0.8052375  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram No stopwords----------------------------------------------
Unigram nsw Val acc at alpha= 0.1  is  0.8072125
Unigram nsw Val acc at alpha= 0.4  is  0.8085875
Unigram nsw Val acc at alpha= 0.5  is  0.80875
Unigram nsw Val acc at alpha= 1.0  is  0.8084
Unigram nsw Val acc at alpha= 1.5  is  0.8081375


Unigrams nsw val Best accuracy= 0.80875  at alpha= 0.5
Unigrams nsw test accuracy= 0.80645  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram stopwords----------------------------------------------
Bigram sw Val acc at alpha= 0.1  is  0.8173875
Bigram sw Val acc at alpha= 0.4  is  0.8203375
Bigram sw Val acc at alpha= 0.5  is  0.81995
Bigram sw Val acc at alpha= 1.0  is  0.818175
Bigram sw Val acc at alpha= 1.5  is  0.81655


Bigrams sw val Best accuracy= 0.8203375  at alpha= 0.4
Bigrams sw test accuracy= 0.8175625  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram No stopwords----------------------------------------------
Bigram nsw Val acc at alpha= 0.1  is  0.7774875
Bigram nsw Val acc at alpha= 0.4  is  0.7839375
Bigram nsw Val acc at alpha= 0.5  is  0.7844875
Bigram nsw Val acc at alpha= 1.0  is  0.784975
Bigram nsw Val acc at alpha= 1.5  is  0.7842125


Bigrams nsw val Best accuracy= 0.784975  at alpha= 1.0
Bigrams nsw test accuracy= 0.7832625  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram stopwords----------------------------------------------
1
2
3
unigram+bigram sw Val acc at alpha= 0.1  is  0.8280375
1
2
3
unigram+bigram sw Val acc at alpha= 0.4  is  0.8299
1
2
3
unigram+bigram sw Val acc at alpha= 0.5  is  0.829425
1
2
3
unigram+bigram sw Val acc at alpha= 1.0  is  0.82755
1
2
3
unigram+bigram sw Val acc at alpha= 1.5  is  0.8261


unigram+bigram sw val Best accuracy= 0.8299  at alpha= 0.4
unigram+bigram sw test accuracy= 0.8271  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram No stopwords----------------------------------------------
unigram+bigram nsw Val acc at alpha= 0.1  is  0.820075
unigram+bigram nsw Val acc at alpha= 0.4  is  0.8256875
unigram+bigram nsw Val acc at alpha= 0.5  is  0.825575
unigram+bigram nsw Val acc at alpha= 1.0  is  0.826475
unigram+bigram nsw Val acc at alpha= 1.5  is  0.8256125


unigram+bigram nsw val Best accuracy= 0.826475  at alpha= 1.0
unigram+bigram nsw test accuracy= 0.824325  at best value of alpha
------------------------------------------------------------------------------------------------ -->













<!-- SklearnClassifier using Nltk -->
<!-- 
 #1. uses zip for format provided i.e [({"hello":True},1),(),()] 
 #2. calls the dict vectorizer through train.
 #3. Then we feed it with sklearn's mnb classifier -->
<!-- 
Naive Bayes classifier follows the conditional independence of each of the features in the model, while Multinomial NB classifier is a specific instance of a NB classifier which uses a multinomial distribution for each of the features.
 
https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c -->
<!-- https://en.wikipedia.org/wiki/Multinomial_distribution
https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52
https://scikit-learn.org/stable/modules/feature_extraction.html
https://datascience.stackexchange.com/questions/31048/pros-cons-of-stop-word-removal#:~:text=If%20you%20are%20using%20some,won't%20drive%20your%20analysis.
https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418

http://www.sfs.uni-tuebingen.de/~keberle/NLPTools/presentations/NLTK/NLTK_Classifiers.pdf

nltk's nltk.classify.scikitlearn accepts vectorized dictionary which can either be a  feature vector with 
with most common words for keys and booleans as values. Or It can also be just the words from the ith 
reviews with True values.  --> 
<!-- 
Package defined by nltk which takes sklearn classifier 

##### class SklearnClassifier(ClassifierI):
##### 

#####     def __init__(self, estimator, dtype=float, sparse=True):
      
#####   self._clf = estimator
      
#####   self._encoder = LabelEncoder()
      
#####   self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse)
    
##### def classify_many(self, featuresets):
     
#####    X = self._vectorizer.transform(featuresets)
      
#####   classes = self._encoder.classes_
      
#####   return [classes[i] for i in self._clf.predict(X)]


##### :param featuresets: An iterable over featuresets, each a dict mapping
   
#####  strings to either numbers, booleans or strings.

##### :return: The predicted class label for each input sample.



#####  def train(self, labeled_featuresets):
      
#####   """
      
#####   Train (fit) the scikit-learn estimator.

      
#####   :param labeled_featuresets: A list of ``(featureset, label)``
      
#####       where each ``featureset`` is a dict mapping strings to either
      
#####       numbers, booleans or strings.
      
#####   """


#####   X, y = list(zip(*labeled_featuresets))
      
#####   X = self._vectorizer.fit_transform(X)
      
#####   y = self._encoder.fit_transform(y)
      
#####   self._clf.fit(X, y)
 
#####   return self --> 

 





<!-- 
 https://stackoverflow.com/questions/40230865/countvectorizer-and-out-of-vocabulary-oov-tokens
Right now I'm using CountVectorizer to extract features. However, I need to count words not seen during fitting.

During transforming, the default behavior of CountVectorizer is to ignore words that were not observed during fitting. But I need to keep a count of how many times this happens!

How can I do this?

Thanks!


There is no inbuilt way in scikit-learn to do this, you need to write some additional code to be able to do this. However you could use the vocabulary_ attribute of CountVectorizer to achieve this.

Cache the current vocabulary
Call fit_transform
Compute the diff with the new vocabulary and the cached vocabulary -->



______________
_____________
____________
____________

### Instructions to Run: 
- To Train the model: `python3 A3/main.py data/` 
- To Test the model: `python3 A3/inference.py data/sample.txt`

### Most similar words: Good

- great --> 0.7551022171974182
- decent --> 0.7508329749107361
- fantastic --> 0.6584885120391846
- nice --> 0.6501882076263428
- wonderful --> 0.6439023017883301
- bad --> 0.6197049617767334
- superb --> 0.6087694764137268
- excellent --> 0.6060928702354431
- terrible --> 0.5937264561653137
- terrific --> 0.5934835076332092
- reasonable --> 0.5878351926803589
- poor --> 0.5779930353164673
- impressive --> 0.5588662624359131
- fair --> 0.5476508140563965
- horrible --> 0.5454399585723877
- okay --> 0.5450385808944702
- awesome --> 0.5421015620231628
- pleasant --> 0.5388332605361938
- strong --> 0.5357953310012817
- amazing --> 0.5348005890846252


### Most similar words: Bad

- horrible --> 0.6506983041763306
- terrible --> 0.6456992626190186
- good --> 0.6197049617767334
- awful --> 0.5941882133483887
- lame --> 0.5502369999885559
- Faced --> 0.5411893129348755
- weak --> 0.5405016541481018
- funny --> 0.5383061766624451
- nasty --> 0.5296658873558044
- poor --> 0.5269413590431213
- obvious --> 0.509117841720581
- fake --> 0.50090092420578
- stupid --> 0.4993632137775421
- strong --> 0.49160268902778625
- lousy --> 0.49101123213768005
- strange --> 0.48473429679870605
- loud --> 0.4782383441925049
- upset --> 0.47385603189468384
- weird --> 0.473296195268631
- overpowering --> 0.46788638830184937

### Are the words most similar to “good” positive, and words most similar to “bad” negative?
- *"Good"*: No, Not all but most of the words most similar to good are positive. Exception: "bad" "terrible" 
- *"Bad"*: No, Not all but most of the words most similar to bad are negative. Exception: "good"
- _Note_: Tried using multiple hyperparameters, similar patterns were seen.

### Analysis: 
1. The most similar words are generated by finding the cosine distance(dot product of vectors) which is ∝ similarity between the words. Word2vec's objective function is trying to maximize the magnitude of dot product of wv and its context wv which determines the similarity. Words like "good" and "bad" are very far apart semantically but are appearing in the same context.
    - Ex: 
        1. It looks nice, but doesn't work like the *good* and oldie!Too *bad*.
        2. But there were more *bad* pots than *good*.
        3. In choosing the Waring model, I researched and read many Amazon reviews, both *good* and *bad*.
        4. The *good*: the non-stick surface works great and is easy to clean.The *bad*: does a terrible job cooking rice.
    - Window size plays an important role while the training phase as the network learns the statistics from the number of times each pairing shows up. In other words, the vector of two words will be pulled closer.
    - So, the similarity is more if the words are closer in the corpus very often which is happening in our case for *good* and *bad* as seen in above examples.
2. The negation or interchangeable usage in the same context can also be one of the reason for the same. Ex: {"good","bad"}, {"cheap","expensive"}, {"sweet","salty"} used interchangeably. Ex:  it's not sweet or salty.


#### References:
- https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/
- http://jalammar.github.io/illustrated-word2vec/
- http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
- https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py#L485

_______________
______________
_______________
_______________
<!-- input:  [batchsize,textsize]-[256,24]
embedding: [batchsize,textsize,vecsize]-(256,24,350)
flatten (Flatten)->(None, 8400)  
Dense(None,120)
Dropout(None,120)
output_dense(None,2) -->
## Running Instructions:
1. main.py => To generate model : 
    `python3 A4/main.py data/`
2. inference.py => To predict txt file using saved model:
    `python3 A4/inference.py data/sample_test.txt tanh`

- Width of the network at H.L
    - neurons: 64 
        - accuracy: 0.7587 | val_accuracy: 0.7427
        - Test Accuracy: 74.24%
    - neurons: 128
        - accuracy: 0.7613 | val_accuracy: 0.7519
        - Test Accuracy: 75.14%
    - Note: Selecting 128 neurons in H.L
- Selecting dataset with stopwords. Almost similar accuracies were seen with both the datasets. Test Accuracy running baseline(No Dropout, No L2) with ReLU.
    - With stopwords: 74.45%
    - Without stopwords: 74.25%
    - Time complexity will be more but It may converge better given the fact, MNB also performed better in this case as there were stopwords contributing to sentiment.

Activation function | L2-norm regularization | Dropout | Train Accuracy(%) | Val Accuracy(%) | Test Accuracy(%)
--- | --- | --- | --- | --- | ---
relu | False | False | 78.94 | 74.23 | 74.28
relu | False | True(0.5) | 73.96 | 74.55 | 74.30    
relu | True(0.01) | True(0.5) | 63.20 | 70.74 | 70.55
relu | True(0.001) | True(0.5) | 65.60 | 69.91 | 69.90
tanh | False |False | 78.25 | 73.14 | 74.47
tanh | False |True(0.5) | 73.12 | 73.63 | 74.98    
tanh | True(0.01) | True(0.5) | 68.18 | 71.36 | 71.32
tanh | True(0.001) | True(0.5) | 70.37|  73.10 | 72.82
sigmoid | False | False | 83.84 | 74.35 | 74.18
sigmoid | False | True(0.5) | 74.84 | 74.99 | 74.25
sigmoid | True(0.01) | True(0.5) | 63.19 | 67.30 | 67.17
sigmoid | True(0.001) | True(0.5) | 68.93 | 72.02 | 71.92

Best Model at 0.2 Dropout after checking on [0.1,0.2,0.3,0.5]:
Activation function | L2-norm regularization | Dropout | Train Accuracy(%) | Val Accuracy(%) | Test Accuracy(%)
--- | --- | --- | --- | --- | ---
relu | False | True(0.2) | 73.11 | 74.76 | 74.45   
tanh | False |True(0.2) | 73.25 | 73.97 | 75.05  
sigmoid | False |True(0.2) | 74.10 | 74.14 | 74.26      

### Note: Accuracies could have been better, if made Trainables=True at embedding layer which drastically increases the time and may lead to overfitting as those weights are updated.

# Analysis
__________________________________________
### Effect of activation functions on results (ReLU,tanh,sigmoid)

- All the three activation functions provide almost similar results with a mean change in accuracy of (+-)1% for all the models.
- tanh works better than ReLU and sigmoid by 1% of accuracy and it took 120s/epoch which is slightly more than the ReLU taking 96s(approx).
        ReLU being a ramp function, doesn't end up with vanishing gradient in case of deep networks whereas, sigmoid functions may end up in vanishing gradient problem i.e. if x<-2(lets say) or x>2, theh derivative becomes small and model may not be able to converge leading to a non firing dead neuron. No problem of dying ReLU was expected or seen. Tested separately using "tf.keras.backend.gradients".
- On the other hand "tanh" works slightly better than the sigmoid in our case giving an accuracy of 75.05% being costly at 120s/epoch. Reason being, the max value of derivative is upto 1.0 for tanh and 0.25 for sigmoid. So the loss was also reduced largely with large updates. 
- Time:
    - ReLU<tanh<sigmoid
- Loss(val):
    - tanh<ReLU<sigmoid (Negligible change)
- Thus, we may choose tanh if accuracy is considered leaving the time complexity behind. ReLU can be  chosen as our Activation function given the time complexity and decent accuracy with no problem of vanishing gradient.
<!-- - Note: Please see the plotted loss at bottom for 5 epochs.  -->
________________________________________________________
 ### Effect of L2-norm regularization
 
- An average decrease of 3-5% accuracy can be seen when using L2-norm in all the cases above.  
- Regularization penalizes the coefficients/wt. matrices to avoid overfitting. Cost function = Loss(cross entropy) + Regularization_term. It reduces the weight matrix close to zero and reduces the overfitting.
- L2 produces small non zero/non sparse coefficients/wts and when L2 used with sigmoid which is not zero averaged/centered, hence the accuracy can be seen dropping for our sparse data.
- whereas model with activation function "tanh" being zero centered isn't getting effected that much on adding l2 reg, although not improving the accuracy as well. 
- However, the training accuracy on adding l2 reg is less than the validation accuracy for all the cases which draws a good pattern of proof.

__________________________________________
### Effect of Dropout
- Adding the dropout of 0.5 increased the validation accuracy leading to less overfitting which can also be seen on comparing the accuracy with and without dropout.
- Dropout is also a regularization technique which randomly selects the n described nodes and removes them with all their incoming and outgoing connections in the network. 
- Thus adding randomness, we are preventing overfitting in our model. Dropout rate of 0.2 was found a decent choice for this data. 



<!-- Typically ridge or ℓ2 penalties are much better for minimizing prediction error rather than ℓ1 penalties. The reason for this is that when two predictors are highly correlated, ℓ1 regularizer will simply pick one of the two predictors. In contrast, the ℓ2 regularizer will keep both of them and jointly shrink the corresponding coefficients a little bit. Thus, while the ℓ1 penalty can certainly reduce overfitting, you may also experience a loss in predictive power. -->

<!-- 
Loss:

![alt text](loss.png "Loss for various Activation functions")
 -->

#### References:
- https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
- https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
 

<!-- 
66% positive relu 90% neg  78
71% positive tanh 88    79.2
67% positive sig 92 79.5 --> 
###### commented!!
