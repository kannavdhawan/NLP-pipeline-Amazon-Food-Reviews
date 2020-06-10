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


outputs:


---------------------------------unigram stopwords----------------------------------------------
Unigram sw Val acc at alpha= 0.01  is  0.80365
Unigram sw Val acc at alpha= 0.1  is  0.806725
Unigram sw Val acc at alpha= 0.5  is  0.8086
Unigram sw Val acc at alpha= 1.0  is  0.808375
Unigram sw Val acc at alpha= 1.5  is  0.80815
Unigram sw Val acc at alpha= 2.0  is  0.807925


Unigrams sw val Best accuracy= 0.8086  at alpha= 0.5
Unigrams sw test accuracy= 0.8078125  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram No stopwords----------------------------------------------
Unigram nsw Val acc at alpha= 0.01  is  0.8013
Unigram nsw Val acc at alpha= 0.1  is  0.80455
Unigram nsw Val acc at alpha= 0.5  is  0.806825
Unigram nsw Val acc at alpha= 1.0  is  0.8068125
Unigram nsw Val acc at alpha= 1.5  is  0.8064
Unigram nsw Val acc at alpha= 2.0  is  0.806075


Unigrams nsw val Best accuracy= 0.806825  at alpha= 0.5
Unigrams nsw test accuracy= 0.8088875  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram stopwords----------------------------------------------
Bigram sw Val acc at alpha= 0.01  is  0.80835
Bigram sw Val acc at alpha= 0.1  is  0.8182625
Bigram sw Val acc at alpha= 0.5  is  0.821
Bigram sw Val acc at alpha= 1.0  is  0.8194625
Bigram sw Val acc at alpha= 1.5  is  0.817475
Bigram sw Val acc at alpha= 2.0  is  0.8156125


Bigrams sw val Best accuracy= 0.821  at alpha= 0.5
Bigrams sw test accuracy= 0.8215  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram No stopwords----------------------------------------------
Bigram nsw Val acc at alpha= 0.01  is  0.7704
Bigram nsw Val acc at alpha= 0.1  is  0.778525
Bigram nsw Val acc at alpha= 0.5  is  0.785725
Bigram nsw Val acc at alpha= 1.0  is  0.784525
Bigram nsw Val acc at alpha= 1.5  is  0.78425
Bigram nsw Val acc at alpha= 2.0  is  0.7833375


Bigrams nsw val Best accuracy= 0.785725  at alpha= 0.5
Bigrams nsw test accuracy= 0.785275  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram stopwords----------------------------------------------
unigram+bigram sw Val acc at alpha= 0.01  is  0.82335
unigram+bigram sw Val acc at alpha= 0.1  is  0.8303125
unigram+bigram sw Val acc at alpha= 0.5  is  0.8310875
unigram+bigram sw Val acc at alpha= 1.0  is  0.8298
unigram+bigram sw Val acc at alpha= 1.5  is  0.8281
unigram+bigram sw Val acc at alpha= 2.0  is  0.8267375


unigram+bigram sw val Best accuracy= 0.8310875  at alpha= 0.5
unigram+bigram sw test accuracy= 0.8317625  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram No stopwords----------------------------------------------
unigram+bigram nsw Val acc at alpha= 0.01  is  0.807425
unigram+bigram nsw Val acc at alpha= 0.1  is  0.818075
unigram+bigram nsw Val acc at alpha= 0.5  is  0.8229875
unigram+bigram nsw Val acc at alpha= 1.0  is  0.8237
unigram+bigram nsw Val acc at alpha= 1.5  is  0.8231
unigram+bigram nsw Val acc at alpha= 2.0  is  0.8226875


unigram+bigram nsw val Best accuracy= 0.8237  at alpha= 1.0
unigram+bigram nsw test accuracy= 0.826275  at best value of alpha

----------------------------------------------------------------------------------------------




---------------------------------unigram stopwords----------------------------------------------
Unigram sw Val acc at alpha= 0.01  is  0.8020375
Unigram sw Val acc at alpha= 0.1  is  0.805575
Unigram sw Val acc at alpha= 0.5  is  0.8069125
Unigram sw Val acc at alpha= 1.0  is  0.8069375
Unigram sw Val acc at alpha= 1.5  is  0.806075
Unigram sw Val acc at alpha= 2.0  is  0.80575


Unigrams sw val Best accuracy= 0.8069375  at alpha= 1.0
Unigrams sw test accuracy= 0.803775  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram No stopwords----------------------------------------------
Unigram nsw Val acc at alpha= 0.01  is  0.80225
Unigram nsw Val acc at alpha= 0.1  is  0.8051125
Unigram nsw Val acc at alpha= 0.5  is  0.8061
Unigram nsw Val acc at alpha= 1.0  is  0.805925
Unigram nsw Val acc at alpha= 1.5  is  0.8052125
Unigram nsw Val acc at alpha= 2.0  is  0.8049375


Unigrams nsw val Best accuracy= 0.8061  at alpha= 0.5
Unigrams nsw test accuracy= 0.8065875  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram stopwords----------------------------------------------
Bigram sw Val acc at alpha= 0.01  is  0.8087625
Bigram sw Val acc at alpha= 0.1  is  0.818525
Bigram sw Val acc at alpha= 0.5  is  0.820275
Bigram sw Val acc at alpha= 1.0  is  0.8184625
Bigram sw Val acc at alpha= 1.5  is  0.8164375
Bigram sw Val acc at alpha= 2.0  is  0.8146


Bigrams sw val Best accuracy= 0.820275  at alpha= 0.5
Bigrams sw test accuracy= 0.8190625  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------Bigram No stopwords----------------------------------------------
Bigram nsw Val acc at alpha= 0.01  is  0.7688625
Bigram nsw Val acc at alpha= 0.1  is  0.7784625
Bigram nsw Val acc at alpha= 0.5  is  0.78445
Bigram nsw Val acc at alpha= 1.0  is  0.7844875
Bigram nsw Val acc at alpha= 1.5  is  0.7838875
Bigram nsw Val acc at alpha= 2.0  is  0.7822375


Bigrams nsw val Best accuracy= 0.7844875  at alpha= 1.0
Bigrams nsw test accuracy= 0.783425  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram stopwords----------------------------------------------
unigram+bigram sw Val acc at alpha= 0.01  is  0.822575
unigram+bigram sw Val acc at alpha= 0.1  is  0.828425
unigram+bigram sw Val acc at alpha= 0.5  is  0.829375
unigram+bigram sw Val acc at alpha= 1.0  is  0.8281625
unigram+bigram sw Val acc at alpha= 1.5  is  0.8266625
unigram+bigram sw Val acc at alpha= 2.0  is  0.8253375


unigram+bigram sw val Best accuracy= 0.829375  at alpha= 0.5
unigram+bigram sw test accuracy= 0.8286625  at best value of alpha
------------------------------------------------------------------------------------------------
---------------------------------unigram+bigram No stopwords----------------------------------------------
unigram+bigram nsw Val acc at alpha= 0.01  is  0.807025
unigram+bigram nsw Val acc at alpha= 0.1  is  0.8187875
unigram+bigram nsw Val acc at alpha= 0.5  is  0.82415
unigram+bigram nsw Val acc at alpha= 1.0  is  0.82445
unigram+bigram nsw Val acc at alpha= 1.5  is  0.82365
unigram+bigram nsw Val acc at alpha= 2.0  is  0.822925


unigram+bigram nsw val Best accuracy= 0.82445  at alpha= 1.0
unigram+bigram nsw test accuracy= 0.8238375  at best value of alpha
------------------------------------------------------------------------------------------------