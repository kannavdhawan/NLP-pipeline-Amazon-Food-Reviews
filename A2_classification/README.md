<!-- ##### https://stackoverflow.com/questions/25155940/nltk-naivebayesclassifier-input-formatting
##### https://www.nltk.org/_modules/nltk/classify/scikitlearn.html

###### Naive Bayes classifier follows the conditional independence of each of the features in the model, while Multinomial NB classifier is a specific instance of a NB classifier which uses a multinomial distribution for each of the features.

References:
# Citations:
### 1. https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
### 2. https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c
### 3. https://en.wikipedia.org/wiki/Multinomial_distribution
https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52
https://scikit-learn.org/stable/modules/feature_extraction.html
https://datascience.stackexchange.com/questions/31048/pros-cons-of-stop-word-removal#:~:text=If%20you%20are%20using%20some,won't%20drive%20your%20analysis.
##### https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418

##### http://www.sfs.uni-tuebingen.de/~keberle/NLPTools/presentations/NLTK/NLTK_Classifiers.pdf

#### nltk's nltk.classify.scikitlearn accepts vectorized dictionary which can either be a  feature vector with 
#### with most common words for keys and booleans as values. Or It can also be just the words from the ith 
#### reviews with True values. 

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
        the test data as negative if that word contributing to negative review is there in any of the rveiew.

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
 unigrams and in unigrams+bigrams.
### Performance comparison:  unigrams, bigrams, unigrams+bigrams:
- Accuracy(unigrams+bigrams)>Accuracy(bigrams)>Accuracy(unigrams) | with stopwords.
    - {unigrams + bigrams} are preserving more information than the others and making the feature vector larger in size as well, so with an increase in the accuracy, we are compromising with the space and time complexity.
- Accuracy(unigrams+bigrams)>Accuracy(unigrams)>Accuracy(bigrams) | without stopwords. 
    - Reason being the same as above comparison, absence of useful words in multiple tokens leading to less accuracy in this set for bigrams.
    - As long as we are increasing the N_grams, accuracy is decreasing(If previous ((N-1) to 1) grams are not included).
- Overall, 
    - If only non essential stopwords are removed(based on the dataset), Accuracy(unigrams+bigrams)>Accuracy(bigrams)>Accuracy(unigrams) for both the sets.


 