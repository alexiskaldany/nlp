""" 
Alexis Kaldany Final Exam

Use BertTokenizer for all encoding/decoding operations.
"""
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd


""" 
Reorganize the data into a format that can be used by the model.
input = 'input'
target = 'target'
"""
dataframe = ""
input_column = ""
target_column = ""

# train_dataframe = 
class CustomDataset(Dataset):
    def __init__(self, dataframe, input_column, target_column, max_len):
        self.dataframe = dataframe
        self.input_column = input_column
        self.target_column = target_column
        self.label_dictionary = {label: i for i, label in enumerate(self.dataframe[self.target_column].unique())}
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenized_input = self.dataframe[self.input_column].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length = self.max_len, truncation = True, padding = "max_length", return_tensors = 'pt')))
        self.one_hot_labels = self.dataframe[self.target_column].apply((lambda x: self.one_hot_encode_label(x)))
        """ 
        Create a one-hot encoding for each label. Turn into a tensor. self.one_hot_labels is a list of tensors.
        """
    def one_hot_encode_label(self, label):
        one_hot = torch.zeros(len(self.label_dictionary))
        one_hot[self.label_dictionary[label]] = 1
        return one_hot
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        return self.tokenized_input[index], self.one_hot_labels[index]


MAX_LEN = 252
HIDDEN_SIZE = 768
NUM_EPOCHS = 1000
PRINT_LOSS_EVERY = 10
train_dataframe = pd.read_csv("/Users/alexiskaldany/school/nlp/TEST_2/covid.csv").rename(columns={"content": "input", "category": "target"})
labels = train_dataframe['target'].unique()
NUM_OF_CLASSES = len(labels)
train_dataset = CustomDataset(train_dataframe, input_column = "input", target_column = "target", max_len = MAX_LEN)
# print(train_dataset[0][0].shape)
# print(train_dataset[0][0].float().squeeze(0).shape)
print(len(train_dataset.tokenized_input))
print(len(train_dataset.one_hot_labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" 
MLP
"""

"""
Create a model that takes in a sequence of tokens and outputs an integer cooresponding to the label.

Use BertTokenizer for encoding.

Create a MLP that takes in the output of the Bert model and outputs a label.
"""

class MLP(nn.Module):
    """ 
    Multi-layer perceptron
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = MLP(MAX_LEN, HIDDEN_SIZE, NUM_OF_CLASSES).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()

""" Train the MLP"""
length_of_dataset = len(train_dataset)
print(length_of_dataset)

for epoch in range(NUM_EPOCHS):
    for i in range (length_of_dataset):
        input, target = train_dataset.__getitem__(i)
        input = input.float().squeeze(0).to(device)
        target = target.to(device)
        output = model(input)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if i % PRINT_LOSS_EVERY == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")
        

"""
RNN
"""

""" 
Transformer
"""
model1_name = 'howey/electra-small-mnli'
