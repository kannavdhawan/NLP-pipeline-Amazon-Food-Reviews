https://stackoverflow.com/questions/25155940/nltk-naivebayesclassifier-input-formatting
https://www.nltk.org/_modules/nltk/classify/scikitlearn.html
https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418
http://www.sfs.uni-tuebingen.de/~keberle/NLPTools/presentations/NLTK/NLTK_Classifiers.pdf

#### nltk's nltk.classify.scikitlearn accepts vectorized dictionary which can either be a  feature vector with 
#### with most common words for keys and booleans as values. Or It can also be just the words from the ith 
#### reviews with True values. 
class SklearnClassifier(ClassifierI):
    def __init__(self, estimator, dtype=float, sparse=True):
        self._clf = estimator
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse)
    def classify_many(self, featuresets):
        X = self._vectorizer.transform(featuresets)
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)]

:param featuresets: An iterable over featuresets, each a dict mapping
    strings to either numbers, booleans or strings.
:return: The predicted class label for each input sample.


 def train(self, labeled_featuresets):
        """
        Train (fit) the scikit-learn estimator.

        :param labeled_featuresets: A list of ``(featureset, label)``
            where each ``featureset`` is a dict mapping strings to either
            numbers, booleans or strings.
        """

        X, y = list(zip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)

        return self


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
------------------------------------------------------------------------------------------------
