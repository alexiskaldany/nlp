# List of word on each line
with open("/Users/alexiskaldany/Desktop/vocab.txt","r") as f:
    vocab_list = f.read().split("\n")

import re

piped_vocab_string = '|'.join(vocab_list)

pattern = re.compile(r"\b({})\b".format(piped_vocab_string),re.IGNORECASE)

test_string = "I am very happy and I am very sadly"

print(pattern.findall(test_string))