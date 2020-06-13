
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
