def stopwords_remover(stopwords_list,List_with_stopwords):
    pass
 # Removing stopwords from both positive and negative reviews
    List_without_stopwords=List_with_stopwords
    for review_type in range(0,len(List_without_stopwords)):
        for list_inner in range(0,len(List_without_stopwords[review_type])):
            for stop in stopwords_list: 
                while stop in List_without_stopwords[review_type][list_inner]:
                    List_without_stopwords[review_type][list_inner].remove(stop)
                # if stop in List_without_stopwords[review_type][list_inner]:
                #     List_without_stopwords[review_type][list_inner].remove(stop)
                # else:
                #     continue
    print(List_without_stopwords[0][0:5])
    print(List_without_stopwords[1][0:5])

    return List_without_stopwords

# def train_val_test(List_with_stopwords,List_without_stopwords):

            