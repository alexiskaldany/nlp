# String and Files
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize

def count_repeated_characters(input_string:str):
    result = {}
    for c in input_string:
        if c in result:
            result[c] += 1
        else:
            result[c] = 1
    return result

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
        
      
# NLTK

def get_lemma(tokens)-> list:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    