#**********************************
import nltk
import string
from spacy.lang.en import English
#**********************************
#==================================================================================================================================================================
# Q1:
# Use the following datframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
#==================================================================================================================================================================

print(20*'-' + 'Begin Q2' + 20*'-')


import pandas as pd
df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                     'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                     'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'],})
occurence_value_counts = df1['Occurrence'].value_counts()
sentiment_value_count = df1['sentiment'].value_counts()
occurence_given_sentiment = df1.groupby(['Occurrence', 'sentiment']).size().unstack(fill_value=0)
occurence_given_sentiment['P'] = occurence_given_sentiment['P']/sentiment_value_count['P']
occurence_given_sentiment['N'] = occurence_given_sentiment['N']/sentiment_value_count['N']
print(occurence_given_sentiment)



print(20*'-' + 'End Q2' + 20*'-')

#==================================================================================================================================================================
# Q2:
# Use the following sentence as a sample text. and Answer the following questions.
# 1- Create binary BOW model by counting and remove stop words
# 2- The code should output the features and the vectors associated with the features.
# 3- How many url is in the text.
#==================================================================================================================================================================
print(20*'-' + 'Begin Q2' + 20*'-')

sentences = [
    "I try to get some features out.",
    "Featues can be represented as vectors.",
    "Vectors are easier to explain.",
]

from nltk.corpus import stopwords

from collections import Counter
from nltk.tokenize import word_tokenize

print(stopwords.words('english'))
sent_tokens = [word_tokenize(sentence) for sentence in sentences]

filtered_sentences = []
for sentence in sent_tokens:
    clean_sentence = []
    for word in sentence:
        if word.lower() not in stopwords.words('english'):
            clean_sentence.append(word)
    filtered_sentences.append(clean_sentence)

sentence_counted = [Counter(sentence) for sentence in filtered_sentences]

dataframe = pd.DataFrame(sentence_counted).fillna(0)
print(dataframe)

### Don't understand "How many url is in the text."

