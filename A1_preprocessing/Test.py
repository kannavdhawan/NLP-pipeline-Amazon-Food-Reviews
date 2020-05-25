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


i=[1,2]
j=i.copy()
print(i+j)