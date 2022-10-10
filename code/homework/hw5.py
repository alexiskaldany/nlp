#%%
# E.1:
# In part of this exercise, you will use regular expression.
import re
# i. Load Email.txt dataset.
with open("/Users/alexiskaldany/school/nlp/code/homework/emails.txt","r") as f:
    emails = f.read()
# ii. Find all email addresses in the text file.
emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",emails)
print(emails[:10])

# iii. Verify the results.
# An email address usually follows these rules:
# • Upper or lower case letters or digits
# • Starting with a letter
# • Followed by a the at sign symbol.
# • Followed by a string of alphanumeric characters. No spaces are allowed
# • Followed by a the dot “.” symbol
# • Followed by a domain extension(e.g.,“com”, “edu”, “net”.)

def verify_email(email):
    if re.match(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",email):
        return True
    else:
        return False
email_verified = [verify_email(email) for email in emails]
# E.2:
# In part of this exercise, you will use regular expression.
# i. Load war and peace by By Leo Tolstoy.

with open("/Users/alexiskaldany/school/nlp/code/homework/war_peace.txt","r") as f:
    war_peace = f.read()
# ii. Check line by line and find any proper name ending with ”..ski” then print them all.
lines = war_peace.split("\n")
print(lines[:10])
full_list = []
names_ending_with_ski = [re.findall(r"[A-Z][a-z]+ski",line) for line in lines]
names_ending_with_ski = [name for name in names_ending_with_ski if name != []]

print(names_ending_with_ski)
#%%
from collections import Counter, defaultdict
# iii. Put all the names into a dictionary and sort them.
names_dict = Counter([name[0] for name in names_ending_with_ski])
sorted_dict = dict(names_dict.most_common(len(names_dict.keys())))
print(sorted_dict)
#%%
# 1
# E.3:
# In part of this exercise, you will use regular expression.
# i. Write a program with regular expression that joins numbers if there is a space between
# them (e.g., ”12 0 mph is a very high speed in the 6 6 interstate.” to ”120 mph is a very
# high speed in the 66 interstate.” )
def join_numbers(text):
    return re.sub(r"(\d) (\d)",r"\1\2",text)

# ii. Write a program with regular expression that find the content in the parenthesise and
# replace it with ”(xxxxx)”
# iii. Write a program that find any word ends with ”ly”.
# iv. Write a program that finds all the quotes in the text and prints the strings in between.
# v. Write a program that finds all words which has 3,4,5 charters in the text.
# v. Write a program that replaces a comma with a hyphen.
# vi. Write a program that extract year, month and date from any url which has date init which
# follows by forward slashes. ”https://www.yahoo.com/news/football/wew/2021/09/02/odell–
# famer-rrrr-on-one-tr-littleball–norman-stupid-author/”
# E.4:
# Answer all the class exercise questions and submit it (Check the instructions).

