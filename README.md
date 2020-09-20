
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
