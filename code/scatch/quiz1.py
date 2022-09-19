from collections import Counter
import pandas as pd
# --------------------------------Q1-------------------------------------------------------------------------------------

with open('sample.txt') as f:
    data = f.read()
    
# print(data)


# --------------------------------Q2-------------------------------------------------------------------------------------

""" Find all sentences that are greater than 15 characters. Hint: The period
identifies a sentence. """

def find_long_sentences(data):
    long_sentences = []
    for sentence in data.split('. '):
        if len(sentence) > 15:
            long_sentences.append(sentence)
    return long_sentences

all_15_char = find_long_sentences(data)

# --------------------------------Q3-------------------------------------------------------------------------------------


""" Find all the non alphabet characters and five most common chars."""

def find_non_alphabet_chars(data):
    non_alphabet_chars = []
    for sentence in data.split('. '):
        for char in sentence:
            if not char.isalpha():
                non_alphabet_chars.append(char)
    return non_alphabet_chars

most_commond_chars = Counter(data).most_common(5)
# --------------------------------Q4-------------------------------------------------------------------------------------
""" For each sentences count the number of five most common chars except
space char."""

def count_five_most_common_chars(data):
    five_most_common_chars = []
    for sentence in data.split('. '):
        five_most_common_chars.append(Counter(sentence).most_common(5))
    return five_most_common_chars



# --------------------------------Q5-------------------------------------------------------------------------------------

""" Save all the sentences in a dataframe (every row contains the sentences of
length more than 15). Create multiple columns that counts the length of the
sentence and the five most common word in each sentence beside all the
counts that you have done in the step4.
"""
import pandas as pd

df = pd.DataFrame(all_15_char, columns=['sentences'])
df['sentence'] = [sentence for sentence in data.split('. ') if len(sentence) > 15]
df['length'] = [len(sentence) for sentence in data.split('. ') if len(sentence) > 15]
df['most_common_1'] = [Counter(sentence).most_common(5)[0] for sentence in data.split('. ') if len(sentence) > 15]
df['most_common_2'] = [Counter(sentence).most_common(5)[1] for sentence in data.split('. ') if len(sentence) > 15]
df['most_common_3'] = [Counter(sentence).most_common(5)[2] for sentence in data.split('. ') if len(sentence) > 15]
df['most_common_4'] = [Counter(sentence).most_common(5)[3] for sentence in data.split('. ') if len(sentence) > 15]
df['most_common_5'] = [Counter(sentence).most_common(5)[4] for sentence in data.split('. ') if len(sentence) > 15]


