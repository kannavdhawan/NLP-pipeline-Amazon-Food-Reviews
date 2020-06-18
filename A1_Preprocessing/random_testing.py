# l_of_l_of_pos_rev=List_with_stopwords[0]
#     train_data_pos=random.sample(l_of_l_of_pos_rev,k=0.8(len(l_of_l_of_pos_rev)))
#     data_left=l_of_l_of_pos_rev-train_data_pos

# x=list(range(0,10))
# print(x)
# import random
# list1=[[1,2],[3,4]]
# # index=random.sample(list1,1)
# # print(list1)
# # rem_list=set(list1)-set(list2)
# # print(rem_list)
# pos_index_list=list(range(0,len(list1)))
# print(pos_index_list)
# lt2=list(list1)
# print(lt2)

# import random
# random.seed(1332)
# pos_index_list=list(range(0,len([1,2,3,4,5,6,7,8])))
#     # indexes of train data pos reviews
# index_train_data_pos=random.sample([1,2,3,4,5,6,7,8],k=int(0.8*(len(pos_index_list))))
# print(index_train_data_pos)

# list1=[[1, 2, 4, 5, 6, 'label'],[2],[3],[4]]
# list2=[[3],[7, 3],[8],[9]]
# a=[]
# for i in list1:
#     a.append(i)
# for j in list2:
#     a.append(j)
# print(a)

# with open('myfile.csv','w') as f:
#     for sublist in a:
#         for item in sublist:
#             f.write(str(item) + ',')
#         f.write('\n')




# list1=[[1, 2, 4, 5, 6, 'label'],[2],[3],[4]]
# for i in list1:
#     i.insert(0,"LABEL")
# print(list1)

# l1=[[1],[2],[3]]
# for i in l1:
#     i.insert(0,'1')
# print(l1[-1:-3])

# l1=[1,2]
# l2=l1.copy()
# l2.remove(2)
# print(l1)
# import pandas as pd 
# a=pd.DataFrame([['abc,cde'],['abc,cde']])
# print(a)
# b=[]
# for i in range(len(a)):
#     b.append(list(a.iloc[i]))
# print(b)

# print('\'')
# print(list('abc,sds'))
# stri='a,b,c'
# print(stri.split(","))


# list33=[['hello']]
# with open('t.csv','w') as f:
#     for i in list33:
#         for j in i:
#             if j.find("'"):
#                 f.write('"'+str(j)+'"')        
#                 print('\"'+str(j)+'\"')
#                 print(True)
#             else:
#                 print(False)



# aa=pd.read_csv("a.csv",sep=';',header=None)
# a=[]
# for i in range(len(aa)):
#     print(aa.iloc[i,0])
#     a=aa.iloc[i,0].split(',')
#     print(a)
#     break


# i=[1,2]
# j=i.copy()
# print(i+j)
# import os
# with open(os.path.join('csv_splits/',"train_sw_labels.csv"),'w') as f:
#     l=[int(1)]*320000+[int(0)]*320000
#     for i in l:
#         f.write(str(i)+'\n')

# b=range(300000)
# a=range(400000)
# x=[index for index in a if index not in b]
# print(x)
# l=[]
# l=[int(1)]*3+[int(0)]*3
# print(l)
# a=[1,2,3]
# b=[4,5,6]
# print(a+b)
# import random
# print(random.sample([1,2,3,4],k=3))
# import os.path
# with open(os.path.join('csv_splits/', "out11.csv"),'w') as f:
#     f.write(str('item')+',')
            # f.write('\''+str(item)+'\''+ ',') # adding string literal to make it easy to convert from csv to list of lists 
                #in the classification task in a2 for making ngrams. basically for making backtracking easy.    f.write('\n')

# def a(): 
#     i=[1,2,3,4]
#     j=[4,5,6,7]
#     k=[333,555]
#     return j,i,k
# b=a()
# print(b)
# print(b[2])


# list1=[]
# for i in ['ABC,DEF','djj,jhj']:
#     list1=i.split(",")
#     print(list1)
# print(list1)


# list_of_things = [21, 2, 93]
# x=[thing for thing in list_of_things]
# print(x)

# import nltk

# from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(stopwords.words('english'))
# import os
# def read_dataset(data_path):
 
    # This method is best for shuffling the dataset by preserving the labels as well. But pandas can do it easily and
    # splitlines is better. 
#     with open(os.path.join(data_path, 'pos.txt')) as f:
#         pos_lines = f.readlines() # reading and inserting \n 
#     with open(os.path.join(data_path, 'neg.txt')) as f:
#         neg_lines = f.readlines()   
#     all_lines = pos_lines + neg_lines   ['I ma kannav\n','I am \n']
#     return list(zip(all_lines, [1]*len(pos_lines) + [0]*len(neg_lines))) #[('hjsj kndjh kshkhd',1),()]

# def test(a):
#     print('Splitting lines in the dataset')
#     all = [line.strip().split() for line in a]   #not possible
#     print(all)
# a=read_dataset('data/')
# test(a)

# print([line.strip().split() for line in ['I am kannav 1','I am kann 0']])
# print(list(zip(['jnjsjhjsd jcdjjd'],[1])))


# from gensim.test.utils import common_texts
# print(common_texts)