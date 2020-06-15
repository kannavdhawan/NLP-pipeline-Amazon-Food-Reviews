
    # ------------------Line 19----------------------------- 
    # L1_train=pd.Series([int(1)]*320000)
    # L0_train=pd.Series([int(0)]*320000)
    # train_labels=pd.concat([L1_train,L0_train],ignore_index=True)
    # train_sw['label']=train_labels
    # train_sw.columns = ['Text', 'Labels']
    # train_nsw['label']=train_labels
    # train_nsw.columns = ['Text', 'Labels']
    # L1_val=pd.Series([int(1)]*40000)
    # L0_val=pd.Series([int(0)]*40000)
    # val_labels=pd.concat([L1_val,L0_val],ignore_index=True)
    # val_sw['label']=val_labels
    # val_sw.columns = ['Text', 'Labels']
    # val_nsw['label']=val_labels
    # val_nsw.columns = ['Text', 'Labels']
    # print(val_sw.info())
    # L1_test=pd.Series([int(1)]*40000)
    # L0_test=pd.Series([int(0)]*40000)
    # test_labels=pd.concat([L1_test,L0_test],ignore_index=True)
    # test_sw['label']=test_labels
    # test_sw.columns = ['Text', 'Labels']
    # test_nsw['label']=test_labels
    # test_nsw.columns = ['Text', 'Labels']



# def dic(j):
#     return dict([(word, True) for word in j]) # returns dictionary
# def data_formatting(input,label_size):
#     temp=[]
#     for i in range(len(input)):
#         if i<=label_size:
#             temp.append((dict([j, True]),1) for j in input[i]) #format-->  [({},1)]
#         else:
#             temp.append((dict([(j, True)]),0) for j in input[i]) #format-->  [({},0)]
#     return temp


# :param labeled_featuresets: A list of ``(featureset, label)``
#             where each ``featureset`` is a dict mapping strings to either
#             numbers, booleans or strings.





# <!-- 
# Naive Bayes classifier follows the conditional independence of each of the features in the model, while Multinomial NB classifier is a specific instance of a NB classifier which uses a multinomial distribution for each of the features.
 
# https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
# https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c -->
# <!-- https://en.wikipedia.org/wiki/Multinomial_distribution
# https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52
# https://scikit-learn.org/stable/modules/feature_extraction.html
# https://datascience.stackexchange.com/questions/31048/pros-cons-of-stop-word-removal#:~:text=If%20you%20are%20using%20some,won't%20drive%20your%20analysis.
# https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418

# http://www.sfs.uni-tuebingen.de/~keberle/NLPTools/presentations/NLTK/NLTK_Classifiers.pdf

# nltk's nltk.classify.scikitlearn accepts vectorized dictionary which can either be a  feature vector with 
# with most common words for keys and booleans as values. Or It can also be just the words from the ith 
# reviews with True values.  --> 
# <!-- 
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

 