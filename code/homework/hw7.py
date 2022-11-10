# =================================================================
# Class_Ex1:
# Lets consider the 2 following senteces
# Sentence 1: I  am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import torch 
print(20*'-' + 'Begin Q1' + 20*'-')
sent_1 = "I am excited about the perceptron network."
sent_2 = "we will not test the classifier with real data."

bag_of_words_1 = sent_1.split() 
bag_of_words_2 = sent_2.split()
full_bag_of_words = bag_of_words_1 + bag_of_words_2
sentiment = [1] * len(bag_of_words_1) + [-1] * len(bag_of_words_2)

input_df = pd.DataFrame({'words': full_bag_of_words, 'sentiment': sentiment})
input_df['sentence'] = np.where(input_df['sentiment'] == 1, 'sent_1', 'sent_2')
print(input_df)

def classify_sentiment(sentence):
    words = sentence.split()
    sentiment = 0
    for word in words:
        sentiment += input_df[input_df['words'] == word]['sentiment'].values[0]
    score = sentiment / len(words)
    return score





print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the same data in Example 1 but instead of hardlim use logsigmoid as transfer function.
# modify your code inorder to classify negative and positive sentences correctly.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')








print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# The following function is given
# F(x) = x1^2 + 2 x1 x2 + 2 x2^2 +x1
# use steepest decent algorithm to find the minimum of the function.
# Plot the the function in 3d and then plot the counter plot with the all the steps.
# use small value as a learning rate.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')









print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Use the following corpus of data
# sent1 : 'This is a sentence one and I want to all data here.',
# sent2 :  'Natural language processing has nice tools for text mining and text classification. I need to work hard and try a lot of exericses.',
# sent3 :  'Ohhhhhh what',
# sent4 :  'I am not sure what I am doing here.',
# sent5 :  'Neural Network is a power method. It is a very flexible architecture'

# Train ADALINE network to find  a relationship between POS (just verbs and nouns) and the the length of the sentences.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')









print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Read the dataset.csv file. This datset is about the EmailSpam.
# Use a two layer network and to classify each email
# You are not allowed to use any NN packages.
# You can use previous NLP packages to read the data process it (NLTK, spaCY)
# Show the classification report and mse of training and testing.
# Try to improve your F1 score. Explain which methods you used.
# Hint. Clean the datset, use all the preprocessing techniques that you learned.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')









print(20*'-' + 'End Q5' + 20*'-')
# =================================================================