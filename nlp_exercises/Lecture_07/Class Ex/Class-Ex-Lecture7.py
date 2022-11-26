# =================================================================
# Class_Ex1:
# Use glove word embeddings to train the MLP of the example
# % --------------------------------------------------------

# 1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/, unzip it and move glove.6B.50d.txt to the
# current working directory.

# 2. Define a function that takes as input the vocab dict from the example and returns an embedding dict with the token
# ids from vocab dict as keys and the 50-dim Tensors from the glove embeddings as values.

# 3. Define a function to return a Tensor that contains the tensors corresponding to the glove embeddings for the tokens
# in our vocabulary. The ones not found on the glove vocabulary are given tensors of 0s. This will happen more often
# than expected because our tokenizer is different than the one used for glove.

# 4. Replace the embedding weights of the model with the loop-up table returned by the function defined in 4. Check some
# of these vectors visually against the glove.6B.50d.txt file to make sure the correct embeddings are being used.

# 6. Add an option to freeze the embeddings so that they are not learnt. This will result in a poor performance because
# there are quite a few tokens which we don't have glove embeddings for (as mentioned in 4.), so we need to learn these.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
import torch
globe_path = "/Users/alexiskaldany/school/nlp/glove.6B.50d.txt"
def return_glove_dict(glove_path):
    with open(globe_path, 'r') as f:
            glove = {line.split()[0]: torch.tensor([float(x) for x in line.split()[1:]]) for line in f}
    return glove

def return_tensor(vocab_list, glove_dict):
    tensor_list = []
    for word in vocab_list:
        if word in glove_dict:
            tensor_list.append(glove_dict[word])
        else:
            tensor_list.append(torch.zeros(50))
    return torch.stack(tensor_list)

# from torch import nn
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import DataLoader
# from torchtext.datasets import AG_NEWS
# import time
# # ------------------------------------------------------------------------------
# train_iter = list(AG_NEWS(split='train'))
# test_iter = list(AG_NEWS(split='test'))
# print(train_iter[0])
# # ------------------------------------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# tokenizer = get_tokenizer('basic_english')
# def yield_tokens(data_iter):
#     for _, text in data_iter:
#         yield tokenizer(text)

# vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])
# print(vocab(['here', 'is', 'an', 'example']))
# # ------------------------------------------------------------------------------
# text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: int(x) - 1
# print(text_pipeline('here is the an example'))
# print(label_pipeline('10'))
# # ------------------------------------------------------------------------------
# def collate_batch(batch):
#     label_list, text_list, offsets = [], [], [0]
#     for (_label, _text) in batch:
#          label_list.append(label_pipeline(_label))
#          processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#          text_list.append(processed_text)
#          offsets.append(processed_text.size(0))
#     label_list = torch.tensor(label_list, dtype=torch.int64)
#     offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
#     text_list = torch.cat(text_list)
#     return label_list.to(device), text_list.to(device), offsets.to(device)
# # ------------------------------------------------------------------------------
# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()
#     def init_weights(self):
#         initrange = 0.5
#         #self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.embedding.weight.data = return_tensor(vocab.get_itos(), return_glove_dict(globe_path))
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()
#     def forward(self, text, offsets):
#         embedded = self.embedding(text, offsets)
#         return self.fc(embedded)
# # ------------------------------------------------------------------------------
# num_class = len(set([label for (label, text) in train_iter]))
# vocab_size = len(vocab)
# emsize = 64
# model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
# # ------------------------------------------------------------------------------
# def train(dataloader):
#     model.train()
#     total_acc, total_count = 0, 0
#     log_interval = 500
#     start_time = time.time()

#     for idx, (label, text, offsets) in enumerate(dataloader):
#         optimizer.zero_grad()
#         predicted_label = model(text, offsets)
#         loss = criterion(predicted_label, label)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
#         optimizer.step()
#         total_acc += (predicted_label.argmax(1) == label).sum().item()
#         total_count += label.size(0)
#         if idx % log_interval == 0 and idx > 0:
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches '
#                   '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
#                                               total_acc/total_count))
#             total_acc, total_count = 0, 0
#             start_time = time.time()

# def evaluate(dataloader):
#     model.eval()
#     total_acc, total_count = 0, 0

#     with torch.no_grad():
#         for idx, (label, text, offsets) in enumerate(dataloader):
#             predicted_label = model(text, offsets)
#             loss = criterion(predicted_label, label)
#             total_acc += (predicted_label.argmax(1) == label).sum().item()
#             total_count += label.size(0)
#     return total_acc/total_count
# # ------------------------------------------------------------------------------
# EPOCHS = 10
# LR = 0.001
# BATCH_SIZE = 64
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# # ------------------------------------------------------------------------------
# train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE,
#                               shuffle=True, collate_fn=collate_batch)
# test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE,
#                               shuffle=True, collate_fn=collate_batch)


# for epoch in range(1, EPOCHS + 1):
#     epoch_start_time = time.time()
#     train(train_dataloader)
#     accu_val = evaluate(test_dataloader)
#     print('-' * 59)
#     print('| end of epoch {:3d} | time: {:5.2f}s | '
#           'valid accuracy {:8.3f} '.format(epoch,
#                                            time.time() - epoch_start_time, accu_val))
#     print('-' * 59)
# # ------------------------------------------------------------------------------
# torch.save(model.state_dict(), 'model_weights.pt')
# # load the model first for inference
# model.load_state_dict(torch.load('model_weights.pt'))
# model.eval()


print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the following corpus
#
corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']
# Train a  two layer neural network and show the the result of word2vec.
# Hint:
# 1- Remove the stop words
# 2- Use binary encoding for each word
# 3- Try a window size of 2 and 3
# 4- Make the embedding size of 2. Plot the each word and explain the results
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy
LR = 1e-2
N_EPOCHS =2000
PRINT_LOSS_EVERY = 1000
EMBEDDING_DIM = 2
# -------------------------------------------------------------------------------------

def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    corpus = [[word for word in sentence.split() if word not in stop_words] for sentence in corpus]
    return corpus
corpus = remove_stop_words(corpus)
print(corpus)
words = []
for text in corpus:
    words.extend(text)
words = set(words)
print(words)

word2int = {}

for i, word in enumerate(words):
    word2int[word] = i
sentences = []
for sentence in corpus:
    sentences.append(sentence)

WINDOW_SIZE = 2
data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0):
            min(idx + WINDOW_SIZE, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
                
for text in corpus:
    print(text)
df = pd.DataFrame(data, columns = ['input', 'label'])
print(df.head(10))
print(df.shape)

ONE_HOT_DIM = len(words)
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = []
Y = []
for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))
X_train = np.asarray(X)
Y_train = np.asarray(Y)

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(12, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 12)
        self.act1 = torch.nn.Softmax(dim=1)
    def forward(self, x):
        out_em = self.linear1(x)
        output = self.linear2(out_em)
        output = self.act1(output)
        return out_em, output
p = torch.Tensor(X_train)
p.requires_grad = True
t = torch.Tensor(Y_train)

model = MLP(EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    _, t_pred = model(p)
    loss = criterion(t, t_pred)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

vectors = model.linear1._parameters['weight'].cpu().detach().numpy().transpose()

print(vectors)

w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = list(words)
w2v_df = w2v_df[['word', 'x1', 'x2']]
print(w2v_df)

fig, ax = plt.subplots()
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))

PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()



print(20*'-' + 'End Q2' + 20*'-')






