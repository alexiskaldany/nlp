import re 

with open("data/Diagrams_ECCV2016.txt", "r") as f:
    text = f.read()
    
# pattern = re.compile(r"\sGod.created")
# matches = pattern.finditer(text)
# [print(match.span(),text[match.span()[0]:match.span()[1]]) for match in matches]
# print(matches)

# pattern_new_line = re.compile(r"\n")
# matches = pattern_new_line.finditer(text)
# for match in matches:
#     print(match.span(),text[match.span()[0]:match.span()[1]])
    
# digit_pattern = re.compile(r"[0-9]")
# matches = digit_pattern.finditer(text)

# pattern_word = re.compile(r"\w+")
# matches = pattern_word.finditer(text)
# match_list = [(match.span(),text[match.span()[0]:match.span()[1]]) for match in matches]

# print(len(match_list))
# print(match_list[:10])

# paper_text = Classical(text)
# word_matches,word_match_list = paper_text.regex_matches(pattern=r"\w+")

# pattern = re.compile(r"\n\B")
# matches = pattern.finditer(text)
# [print(match.span(),text[match.span()[0]:match.span()[1]]) for match in matches]

