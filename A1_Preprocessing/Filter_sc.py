import numpy as np 
# Removing separated special characters which are not joined with any other word-- 
def spaced_special_char_filter(spec_char_list,ob_tokenizer):
    print("Just checking the size using numpy explicitly/ No use of numpy or numpy array elsewhere: ",np.array(ob_tokenizer).shape) # just checking the shape, nothing else. 
    pass
    partially_filtered_positive_reviews=[]
    partially_filtered_negative_reviews=[]
    partial_filtered=[]
#------ Filter "Explicitly spaced" special characters on positive reviews -------
    for pos_list in ob_tokenizer[0]:
        innerlist_pos=[]
        pass
        # print(i)
        # break
        for word in pos_list:
            if word not in spec_char_list:
                innerlist_pos.append(word)
            else:
                continue
        partially_filtered_positive_reviews.append(innerlist_pos)
    # print(partially_filtered_positive_reviews[0:3])

#------- Filter "Explicitly spaced" special characters on negative reviews --------
    for neg_list in ob_tokenizer[1]:
        innerlist_neg=[]
        for word in neg_list:
            if word not in spec_char_list:
                innerlist_neg.append(word)
            else:
                continue
        partially_filtered_negative_reviews.append(innerlist_neg)
    
    partial_filtered.append(partially_filtered_positive_reviews)
    partial_filtered.append(partially_filtered_negative_reviews)
    print("partially filtered Negative reviews testing: ", partially_filtered_negative_reviews[9])
    return partial_filtered

# Removing symbols concatinated with other strings
def spec_char_filter(spec_char_list,ob_tokenizer):
    final_positive_tokens=[]
    final_negative_tokens=[]
    final_tokens=[]


    for pos_list in ob_tokenizer[0]: # ['','',']
        new_list=[] # Remaking above list  
        for word in pos_list:
            val=[] # temp list for each word
            for character in word:
                if character in spec_char_list:
                    continue
                else:
                    val.append(character)
            string=""
            string=string.join(val)
            new_list.append(string)
        final_positive_tokens.append(new_list)
    print("random testing------------------------------")
    print("Testing final positive tokens: ",final_positive_tokens[9])

    for neg_list in ob_tokenizer[1]: # ['','',']
        new_list1=[] # Remaking above list  
        for word in neg_list:
            val1=[] # temp list for each word
            for character in word:
                if character in spec_char_list:
                    continue
                else:
                    val1.append(character)
            string=""
            string=string.join(val1)
            new_list1.append(string)
        final_negative_tokens.append(new_list1)
    print("random testing------------------------------")
    print("Testing final positive tokens: ",final_negative_tokens[9])

    final_tokens.append(final_positive_tokens)
    final_tokens.append(final_negative_tokens)
    
    print("--Testing final object for first five values of positive reviews--")
    print(final_tokens[0][0:5])
    
    print("--Testing final object for first five values of negative reviews--")
    print(final_tokens[1][0:5])
    return final_tokens