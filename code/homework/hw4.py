#%%
# E.1:
# In part of this exercise, you will use Spacy software to explore the unrealistic news dataset
# called data.txt.
import pandas as pd
# i. Use pandas to load the data.csv data.

df_2 = pd.read_csv("/Users/alexiskaldany/school/nlp/code/homework/data (1).csv")
# ii. Use Spacy to find the word level attributions ( Tokenized word, StartIndex, Lemma,
# punctuation,white space ,WordShape, PartOfSpeech, POSTag). Use one of the titles
# in the dataframe and create a dataframe which rows are the word and the columns are
# attributions.
import spacy
nlp = spacy.load('en_core_web_sm')
target =df_2['title'][0]
print(target)
tokens = nlp(target)
StartIndex = [tokens[i].idx for i in range(len(tokens))]
Lemma = [tokens[i].lemma_ for i in range(len(tokens))]
Punctuation = [tokens[i].is_punct for i in range(len(tokens))]
white_space = [tokens[i].is_space for i in range(len(tokens))]
word_shape = [tokens[i].shape_ for i in range(len(tokens))]
partofspeech = [tokens[i].pos_ for i in range(len(tokens))]
posttag = [tokens[i].tag_ for i in range(len(tokens))]
new_df = pd.DataFrame({'Tokenized word':tokens,'StartIndex':StartIndex,'Lemma':Lemma,'Punctuation':Punctuation,'white space':white_space,'WordShape':word_shape,'PartOfSpeech':partofspeech,'POSTag':posttag})
print(new_df.head)

#%%
# iii. Use spacy and find entities on the text in part ii.
entities = [tokens[i].ent_type_ for i in range(len(tokens))]


# iv. Grab a different title and use spacy to chunk the noun phrases, label them and finally find
# the roots of each chunk.
target_2 = df_2['title'][1]
noun_phrase = [chunk.text for chunk in nlp(target_2).noun_chunks]
roots = [chunk.root.text for chunk in nlp(target_2).noun_chunks]

# v. Use SPacy to analyzes the grammatical structure of a sentence, establishing relationships
# between ”head” words and words which modify those heads. Hint: Insatiate the nlp doc
# and then look for text, dependency, head text, head pos and children of it.

doc = nlp(target_2)
text = [token.text for token in doc]
dependency = [token.dep_ for token in doc]
head_text = [token.head.text for token in doc]
head_pos = [token.head.pos_ for token in doc]
children = [list(token.children) for token in doc]
# vi. Use spacy to find word similarly measure. Spacy has word vector model as well. So we
# can use the same to find similar words. Use spacy large model to get a decent results.
nlp = spacy.load('en_core_web_lg')
word_similarity = nlp('cat').similarity(nlp('dog'))

# E.2:
# In part of this exercise, you will use Spacy software to explore the tweets dataset called
# data1.txt.

# i. Use pandas to load the data1.csv data.
df_1 = pd.read_csv('/Users/alexiskaldany/school/nlp/code/homework/data1.csv')
# ii. Let’s look at some examples of real world sentences. Grab a tweet and explain the text
# entities.
print(df_1.head())
tweet = df_1['text'][0]
text = [token.text for token in nlp(tweet)]
# iii. One simple use case for NER is redact names. This is important and quite useful. Find a
# tweet which has a name in it and then redact it by word [REDACTED].
entity_list = [token.ent_type_ for token in nlp(tweet)]
redacted_tweet = [token.text if token.ent_type_ != 'PERSON' else '[REDACTED]' for token in nlp(tweet)]

# E.3:
# Use spacy to answer all the following questions.

# i. Apply part of speech Tags methods in spacy on a sentence.
pos_tag = [token.pos_ for token in nlp(tweet)]
# ii. Apply syntactic dependencies methods in spacy on same sentence that you used on part i.
syntactic_dependencies = [token.dep_ for token in nlp(tweet)]
# iii. Apply named entities methods in spacy on the following sentence ”Apple is looking at
# buying U.K. startup for 1 billion dollar”.
string= 'Apple is looking at buying U.K. startup for 1 billion dollar'
entities = [token.ent_type_ for token in nlp(string)]
# iv. Apply document similarity on two separate document (2 sentences).
similarity = nlp(tweet).similarity(nlp('Apple is looking at buying U.K. startup for 1 billion dollar'))
# E.4:
# Answer all the class exercise questions and submit it (Check the instructions).

# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
import re
print(20*'-' + 'Begin Q1' + 20*'-')

def check_string(string):
    pattern = re.compile(r'[^a-zA-Z0-9.]')
    string = pattern.sub('', string)
    return string








print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Write a function that matches a string in which a followed by zero or more b's.
# Sample String 'ac', 'abc', abbc'
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')










print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Write Python script to find numbers between 1 to 3 in a given string.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')









print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Write a Python script to find the a position of the substrings within a string.
# text = 'Python exercises, JAVA exercises, C exercises'
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')









print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Write a Python script to find if two strings from a list starting with letter 'C'.
# words = ["Cython CHP", "Java JavaScript", "PERL S+"]
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')









print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# Write a Python script to remove everything except chars and digits from a string.
# USe sub method
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')









print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Scrape the the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and pint couple of the sentences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')









print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common words.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')











print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Write a python script that accepts any string and do the following.
# 1- Tokenize the text
# 2- Doe word extraction and clean a text. Use regular expression to clean a text.
# 3- Generate BOW
# 4- Vectorized all the tokens.
# 5- The only package you can use is numpy and re.
# all sentences = ["sentence1", "sentence2", "sentence3",...]
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')










print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Grab any text (almost a paragraph) from Wikipedia and call it text
# Preprocessing the text data (Normalize, remove special char, ...)
# Find total number of unique words
# Create an index for each word.
# Count number of the owrds.
# Define a function to calculate Term Frequency
# Define a function calculate Inverse Document Frequency
# Combining the TF-IDF functions
# Apply the TF-IDF Model to our text
# you are allowed to use just numpy and nltk tokenizer
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')










print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Grab arbitrary paragraph from any website.
# Creat  a list of stopwords manually.  Example :  stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
# Create a list of ignore char Example: ' :,",! '
# Write a LSA class with the following functions.
# Parse function which tokenize the words lower cases them and count them. Use dictionary; keys are the tokens and value is count.
# Clac function that calculate SVD.
# TFIDF function
# Print function which print out the TFIDF matrix, first 3 columns of the U matrix and first 3 rows of the Vt matrix
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')









print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Use the following doc
# doc = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer" ]
# Calculate the binary BOW.
# Use LSA method and distinguish two different topic from the document. Sent 1,2 is about OpenAI and sent3, 4 is about ML.
# Use pandas to show the values of dataframe and lsa components. Show there is two distinct topic.
# Use numpy take the absolute value of the lsa matrix sort them and use some threshold and see what words are the most important.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')







print(20*'-' + 'End Q12' + 20*'-')
# =================================================================














