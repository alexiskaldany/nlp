# ===========================Packages======================================
import spacy 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------
corp = [
    "This is a sentence one and this is a sample text.",
    "Natural language processing has nice tools for text mining and text classification. I need to work hard and try to learn it by doing exericses.",
    "Ohhhhhh what",
    "I am not sure what I am doing is right.",
    "Neural Network is a powerful method. It is a very flexible architecture",
]

# 1. Find a length of sentences.
sentence_length = [len(sentence.split()) for sentence in corp]
print(sentence_length)

# 2. Find all nouns and verbs in the sentences.

nlp = spacy.load('en_core_web_sm')
sentence_nouns = []
for sentence in corp:
    doc = nlp(sentence)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    sentence_nouns.append(nouns)
print(sentence_nouns)


sentence_verbs = []

for verb in corp:
    doc = nlp(verb)
    verbs = [token.text for token in doc if token.pos_ == 'VERB']
    sentence_verbs.append(verbs)
sentence_noun_length = [len(noun) for noun in sentence_nouns]
sentence_verb_length = [len(verb) for verb in sentence_verbs]

dataframe = pd.DataFrame({'corp':corp,'sentence_length': sentence_length, 'sentence_noun_length': sentence_noun_length, 'sentence_verb_length': sentence_verb_length})

all_nouns = list(set([noun for nouns in sentence_nouns for noun in nouns]))
all_verbs = list(set([verb for verbs in sentence_verbs for verb in verbs]))
print(all_nouns)
print(all_verbs)
# 3. Train an ADALINE network and plot the SSE.


def tokenize(sentence, all_nouns, all_verbs):
    # Tokenize the sentence
    doc = nlp(sentence)
    # Initialize the vector
    vector = np.zeros(len(all_nouns) + len(all_verbs))
    # Iterate over the tokens
    # Each token is a word
    # 1 if token == 'noun' or token == 'verb'
    # 0 otherwise (already initialized to 0)
    for token in doc:
        if token.pos_ == 'NOUN':
            vector[all_nouns.index(token.text)] = 1
        elif token.pos_ == 'VERB':
            vector[all_verbs.index(token.text)] = 1
    return vector
def adaline(X, y, learning_rate, epochs):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0
    # Initialize the list of SSE

    sse = []
    for epoch in range(epochs):
        # Initialize the SSE
        sse_epoch = 0
        for i in range(X.shape[0]):
            # Compute the output
            output = np.dot(X[i], weights) + bias
            # Compute the error
            error = y[i] - output
            # Update the weights
            weights += learning_rate * error * X[i]
            # Update the bias
            bias += learning_rate * error
            # Update the SSE
            sse_epoch += error**2
        # Append the SSE
        sse.append(sse_epoch)
    return weights, bias, sse

def plot_sse(sse):
    # Plot the SSE
    plt.plot(sse)
    # Show the plot
    plt.show()
    
    
### Adaline for tokenized arrays
tokenized_arrays = [tokenize(sentence,all_nouns,all_verbs) for sentence in corp]
X = np.stack(tokenized_arrays)
weights, bias, sse = adaline(X, sentence_length, 0.01, 100)
# plot_sse(sse)

# 4. Find the relationship between nouns, verbs and the length of sentences.

number_of_nouns = dataframe['sentence_noun_length']
number_of_verbs = dataframe['sentence_verb_length']

input_array = np.array([(number_of_nouns[i], number_of_verbs[i]) for i in range(len(number_of_nouns))])

weights, bias, sse = adaline(input_array, sentence_length, 0.01, 100)

print(weights)
print(f"The noun count is multiplied by {weights[0]} and the verb count is multiplied by {weights[1]}")
print(f"The bias is {bias}")
print("As the weight for noun count is positive, the more nouns there are, the longer the sentence will be.")
print("The weight for verb count is negative, so the more verbs there are, the shorter the sentence will be.")
# plot_sse(sse)
