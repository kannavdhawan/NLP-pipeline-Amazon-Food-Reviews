##### https://stackoverflow.com/questions/25155940/nltk-naivebayesclassifier-input-formatting
##### https://www.nltk.org/_modules/nltk/classify/scikitlearn.html

###### Naive Bayes classifier follows the conditional independence of each of the features in the model, while Multinomial NB classifier is a specific instance of a NB classifier which uses a multinomial distribution for each of the features.

References:
# Citations:
### 1. https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
### 2. https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c
### 3. https://en.wikipedia.org/wiki/Multinomial_distribution


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
 
#####   return self







| stopwords Removed  | Text features | Accuracy(Test set)|
| ------------------ | ------------- |-------------------|
|        Yes         |    unigrams   |       80.64%      | 
|        Yes         |    bigrams    |       78.32%      | 
|        Yes         |    uni+bi     |       82.43%      | 
|        No          |    unigrams   |       80.52%      | 
|        No          |    bigrams    |       81.75%      | 
|        No          |    uni+bi     |       82.71%      | 

outputs:

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
unigram+bigram sw Val acc at alpha= 0.1  is  0.8280375
unigram+bigram sw Val acc at alpha= 0.4  is  0.8299
unigram+bigram sw Val acc at alpha= 0.5  is  0.829425
unigram+bigram sw Val acc at alpha= 1.0  is  0.82755
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
------------------------------------------------------------------------------------------------


2596 words with don't in test_sw.csv which are removed in test_nsw.csv

<!-- my,iphone,does,feel,protected,and,i,don't,have,to,worry,as,much,about,dropping,my,new,phone. --> pos review
This is a positive review and I have taken ""don't"" as a stopword. If I remove that stopword,
the review will tend to become negative. 
<!-- i,also,like,the,fact,that,i,don't,have,to,worry,about,maintaining,a,fancy,handle. -->--> pos review
remove don't it will be classified as neg
<!-- i,don't,see,it,as,a,big,problem,unless,for,some,reason,you,are,in,a,huge,rush. --> pos review


