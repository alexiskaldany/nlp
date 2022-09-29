#E.1:
# In part of this exercise, you will use nltk to explore the Moby Dick text.
# i. Analyzing Moby Dick text. Load the moby.txt file into python environment. (Load the
# raw data or Use the NLTK Text object)
from nltk.book import text1
import nltk
nltk.download('twitter_samples')

# ii. Tokenize the text into words. How many tokens (words and punctuation symbols) are in
# it?
text1_tokens = text1.tokens
num_of_tokens=len(text1_tokens)
print(num_of_tokens)


# iii. How many unique tokens (unique words and punctuation) does the text have?
unique_tokens = set(text1_tokens)
print(len(unique_tokens))
# iv. After lemmatizing the verbs, how many unique tokens does it have?
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
verb_lemma = [lemmatizer.lemmatize(token, pos='v') for token in text1_tokens]
unique_verb_lemma = set(verb_lemma)
print(len(unique_verb_lemma))
# v. What is the lexical diversity of the given text input?
diversity = len(text1_tokens)/len(unique_tokens)
print(f"lexical diversity:{diversity}")
# vi. What percentage of tokens is ’whale’or ’Whale’?
num_of_whale = text1_tokens.count('whale') + text1_tokens.count('Whale')
percentage = num_of_whale/len(text1_tokens)
print(f"percentage of whale:{percentage}")
# vii. What are the 20 most frequently occurring (unique) tokens in the text? What is their
# frequency?
from collections import Counter
twemty_most_common = Counter(text1_tokens).most_common(20)
print(twemty_most_common)
frequency_of_twemty_most_common = [i[1] for i in twemty_most_common]
print(frequency_of_twemty_most_common)  
# viii. What tokens have a length of greater than 6 and frequency of more than 160?
greater_than_6_freq_more_than_160 = set([i for i in set(text1_tokens) if len(i)>6 and text1_tokens.count(i)>160])
print(greater_than_6_freq_more_than_160)

# ix. Find the longest word in the text and that word’s length.
longest_word = max(text1_tokens, key=len)
print(f"The longest word is :< {longest_word} > and is {len(longest_word)} characters long")
# x. What unique words have a frequency of more than 2000? What is their frequency?
unique_freq_more_than_2000 = set([i for i in set(text1_tokens) if text1_tokens.count(i)>2000])
print(unique_freq_more_than_2000)
print([text1_tokens.count(i) for i in unique_freq_more_than_2000])
# xi. What is the average number of tokens per sentence?
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text1.__str__())
num_of_tokens_per_sentence = [len(i) for i in sentences]
mean_number_of_tokens_per_sentence = sum(num_of_tokens_per_sentence)/len(num_of_tokens_per_sentence)
print(f"The mean number of tokens per sentence: {mean_number_of_tokens_per_sentence}")
# xii. What are the 5 most frequent parts of speech in this text? What is their frequency?
import nltk
most_freq_pos = Counter([i[1] for i in nltk.pos_tag(text1_tokens)]).most_common(5)
print(most_freq_pos)

# E.2:
# Lets get some text file from the Benjamin Franklin wiki page.
# i. Write a function that scrape the web page and return the raw text file.
import requests 
request = requests.get("https://en.wikipedia.org/wiki/Benjamin_Franklin")
text = request.text
with open("./data/benjamin_franklin_raw.txt", "w") as f:
    f.write(text)
# ii. Use BeautifulSoup to get text file and clean the html file.
import bs4 as bs
from lxml import etree
soup = bs.BeautifulSoup(text, 'lxml')
clean_text = soup.get_text()
with open("./data/benjamin_franklin_clean.txt", "w") as f:
    f.write(clean_text)
# iii. Write a function called unknown, which removes any items from this set that occur in the
# Words Corpus (nltk.corpus.words).
def remove_known_words(text):
    tokens = nltk.word_tokenize(text)
    from nltk.corpus import words
    words = set(words.words())
    return [i for i in set(tokens) if i not in words]

# iv. Fins a list of novel words.
list_of_novel_words = remove_known_words(clean_text)
print(len(set(list_of_novel_words)))
print(list_of_novel_words[:25])

# v. Use the porter stemmer to stem all the items in novel words the go through the unknown
# function, saving the result as novel-stems.
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stems_novel_words = [ps.stem(i) for i in list_of_novel_words]
print(stems_novel_words[:25])
# vi. Find as many proper names from novel-stems as possible, saving the result as propernames.
propernames = [i for i in stems_novel_words if i[0].isupper()]
print(len(propernames))




# E.3:
# In part of this exercise, you will use the twitter data.
from nltk.corpus import twitter_samples
print(twitter_samples.__str__())
# i. Load the data and view the first few sentences.
# ii. Split data into sentences using ”\n” as the delimiter.
# iii. Tokenize sentences (split a sentence into a list of words). Convert all tokens into lower
# case so that words which are capitalized
# iv. Split data into training and test sets.
# v. Count how many times each word appears in the data.



# E.4:
# Answer all the class exercise questions and submit it (Check the instructions)."""