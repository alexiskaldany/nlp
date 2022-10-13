# Topics

## NLTK

## SpaCy

```
import spacy
nlp = spacy.load('en_core_web_lg')
```


## RegEx

### Character Classes
\w	Match a single word character a-z, A-Z, 0-9, and underscore (_)

\d	Match a single digit 0-9

\s	Match whitespace including \t, \n, and \r and space character

.	Match any character except the newline

\W	Match a character except for a word character

\D	Match a character except for a digit

\S	Match a single character except for a whitespace character


### Assertions

^   Matches start of input

$   Matches end of input

\b  Matches a word boundary

[xyz] Matches any of the enclosed characters

[a-c]

[^xyz] Matches anything not enclosed

### Quantifiers

*   0 or more times

+   1 or more times

?   0 or 1 times

{n} matches n many times

x{n,}   matches at least "n" occurences of x

x{n,m}  matches at least n many times, at most m many times

x*?           ? after quantifier makes quantifier non-greedy
x+?
x??
x{n}?
x{n,}?
x{n,m}?

## Naive Bayes

## LSA