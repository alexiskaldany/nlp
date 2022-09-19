"""
E.1:
Write a python script that reads a string from the user input and print the following
i. Number of uppercase letters in the string.
ii. Number of lowercase letters in the string
iii. Number of digits in the string
iv. Number of whitespace characters in the string
"""



input_string = "Hello World"

print("Number of uppercase letters: ", sum(1 for c in input_string if c.isupper()))
""" 
E.2:
Write a python script that accepts a string then create a new string by shifting one position to
left.
Example: input : class 2021 output: lass 2021c
"""
# input_string = input("Enter a string: ")
print(input_string[1:] + input_string[0])

"""
E.3:
Write a python script that a user input his name and program display its initials.
Hint: Assuming, user always enter first name, middle name and last name.
"""
# input_string = input("Enter your name: ")
print("Initials: ", input_string[0] + input_string[input_string.find(" ") + 1] + input_string[input_string.rfind(" ") + 1])
""" 
E.4:
Write a python script that accepts a string to setup a passwords. The password must have the
following requirements
• The password must be at least eight characters long.
• It must contain at least one uppercase letter.
• It must contain at least one lowercase letter.
• It must contain at least one numeric digit.
"""

def password_check(password:str):
    result = len(password) >= 8 and any(c.isupper() for c in password) and any(c.islower() for c in password) and any(c.isdigit() for c in password)
    return result
print(password_check("Password1")) # True
print(password_check("password1")) # False


""" 
E.5:
Write a python script that reads a given string character by character and count the repeated
characters then store it by length of those character(s).
"""

def count_repeated_characters(input_string:str):
    result = {}
    for c in input_string:
        if c in result:
            result[c] += 1
        else:
            result[c] = 1
    return result

""" 
E.6:
Write a python script to find all lower and upper case combinations of a given string.
Example: input: abc output: ’abc’, ’abC’, ’aBc’, ...
"""
def lower_upper_case_combinations(input_string:str):
    result = []
    for i in range(2 ** len(input_string)):
        result.append("".join([input_string[j].upper() if (i & (1 << j)) else input_string[j].lower() for j in range(len(input_string))]))
    return result


""" 
E.7:
Write a python script that
i. Read first n lines of a file.
ii. Find the longest words.
iii. Count the number of lines in a text file.
iv. Count the frequency of words in a file.
Hint: first create a test.txt file and dump some textual data in it. Then test your code
"""

def read_first_n_lines(file_name:str, n:int):
    with open(file_name, "r") as f:
        n_lines = f.readlines()[:n]
        combined_lines = "".join(n_lines)
        print(max(combined_lines.split(), key=len))
        results = {}
        for x in combined_lines.split():
            if x in results.keys():
                results[x] += 1
            else:
                results[x] = 1
        print(results)
        
      
read_first_n_lines("/Users/alexiskaldany/school/nlp/data/genesis_1.txt", 20)
  



"""
E.8:
Answer all the class exercise questions and submit it (Check the instructions).
"""
from nltk.book import text4
import nltk
# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
# print(text4.dispersion_plot(["democracy", "freedom", "duties", "America"]))





# print(20*'-' + 'End Q1' + 20*'-')

# # =================================================================
# # Class_Ex2:
# # What is the difference between the following two lines of code? Explain in details why?
# # Make up and example base don your explanation.
# # Which one will give a larger value? Will this be the case for other texts?
# # 1- sorted(set(w.lower() for w in text1))
# # 2- sorted(w.lower() for w in set(text1))
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q2' + 20*'-')








# print(20*'-' + 'End Q2' + 20*'-')
# # =================================================================
# # Class_Ex3:
# # Find all the four-letter words in the Chat Corpus (text5).
# # With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
# #
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q3' + 20*'-')








# print(20*'-' + 'End Q3' + 20*'-')
# # =================================================================
# # Class_Ex4:
# # Write expressions for finding all words in text6 that meet the conditions listed below.
# # The result should be in the form of a list of words: ['word1', 'word2', ...].
# # a. Ending in ise
# # b. Containing the letter z
# # c. Containing the sequence of letters pt
# # d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q4' + 20*'-')









# print(20*'-' + 'End Q4' + 20*'-')
# # =================================================================
# # Class_Ex5:
# #  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
# #  Count occurrences of men, women, and people in each document.
# #  What has happened to the usage of these words over time?
# # Since there would be a lot of document use every couple of years.
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q5' + 20*'-')









# print(20*'-' + 'End Q5' + 20*'-')

# # =================================================================
# # Class_Ex6:
# # The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# # How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
# #
# #
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q6' + 20*'-')








# print(20*'-' + 'End Q6' + 20*'-')
# # =================================================================
# # Class_Ex7:
# # What percentage of noun synsets have no hyponyms?
# # You can get all noun synsets using wn.all_synsets('n')
# #
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q7' + 20*'-')








# print(20*'-' + 'End Q7' + 20*'-')
# # =================================================================
# # Class_Ex8:
# # Write a program to find all words that occur at least three times in the Brown Corpus.
# # USe at least 2 different method.
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q8' + 20*'-')









# print(20*'-' + 'End Q8' + 20*'-')
# # =================================================================
# # Class_Ex9:
# # Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# # Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# # Did you find any strange word in the list? If yes investigate the cause?
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q9' + 20*'-')











# print(20*'-' + 'End Q9' + 20*'-')
# # =================================================================
# # Class_Ex10:
# # Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# # Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q10' + 20*'-')









# print(20*'-' + 'End Q10' + 20*'-')
# # =================================================================
# # Class_Ex11:
# #  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
# #  with all HTML markup removed. Use from urllib import request and
# #  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q11' + 20*'-')








# print(20*'-' + 'End Q11' + 20*'-')
# # =================================================================
# # Class_Ex12:
# # Read in some text from a corpus, tokenize it, and print the list of all
# # wh-word types that occur. (wh-words in English are used in questions,
# # relative clauses and exclamations: who, which, what, and so on.)
# # Print them in order. Are any words duplicated in this list,
# # because of the presence of case distinctions or punctuation?
# # Note Use: Gutenberg('bryant-stories.txt')
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q12' + 20*'-')










# print(20*'-' + 'End Q12' + 20*'-')
# # =================================================================
# # Class_Ex13:
# # Write code to access a  webpage and extract some text from it.
# # For example, access a weather site and extract  a feels like temprature..
# # Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q13' + 20*'-')








# print(20*'-' + 'End Q13' + 20*'-')
# # =================================================================
# # Class_Ex14:
# # Use the brown tagged sentes corpus news.
# # make a test and train sentences and then  use bigram tagger to train it.
# # Then evlaute the trained model.
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q14' + 20*'-')













# print(20*'-' + 'End Q14' + 20*'-')

# # =================================================================
# # Class_Ex15:
# # Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q15' + 20*'-')







# print(20*'-' + 'End Q15' + 20*'-')

# # =================================================================
# # Class_Ex16:
# # Write programs to process the Brown Corpus and find answers to the following questions:
# # 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# # ----------------------------------------------------------------
# print(20*'-' + 'Begin Q16' + 20*'-')





# print(20*'-' + 'End Q16' + 20*'-')


