# Assignment 1
-------------------------------------------------
## index.py: 
#### Data is loaded and all the modules are called over here.
#### Calls -> Functions in Tokenizer, Filter_sc 
## Tokenizer.py:
#### * Input: pos, neg text files
#### * Sentence tokenization. 
#### * Word tokenization. 
#### * Returns a list tokens consisting of [tokenized_pos,tokenized_neg].
## Filter_sc.py:
#### * Input: special_char_list, ob_tokenizer which consists of [tokenized_pos,tokenized_neg] generated in previous step.
#### * A function "spaced_special_char_filter" which removes separated special characters which are not joined with any other word for both the positive and negative tokens.
#### * A function "spec_char_filter" accepting special character list and ob_parially_filtered which is the list of partially filtered tokens from the previous step and this removes the special characters which are concatinated with other characters For instance: '@#method%^' => outputs 'method'.
#### * Returns a list final_tokens consisting of filtered positive and negative tokens without special characters. 


###### Naive Bayes classifier follows the conditional independence of each of the features in the model, while Multinomial NB classifier is a specific instance of a NB classifier which uses a multinomial distribution for each of the features.

References:
# Citations:
### 1. https://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
### 2. https://medium.com/@theflyingmantis/text-classification-in-nlp-naive-bayes-a606bf419f8c
### 3. https://en.wikipedia.org/wiki/Multinomial_distribution
