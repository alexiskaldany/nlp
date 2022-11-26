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
    def __init__(self, dataframe, input_column, target_column,max_len):
        self.dataframe = dataframe
        self.input_column = input_column
        self.target_column = target_column
        self.label_dictionary = {label: i for i, label in enumerate(self.dataframe[self.target_column].unique())}
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenized_input = self.dataframe[self.input_column].apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length = self.max_len, truncation = True, padding = True, return_tensors = 'pt')))
        self.get_one_hot_labels_list = lambda label: [1 if i == self.label_dictionary[label] else 0 for i in range(len(self.label_dictionary))]
        """ 
        Create a one-hot encoding for each label. Turn into a tensor. self.one_hot_labels is a list of tensors.
        """
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        return self.tokenized_input[index], torch.tensor(self.get_one_hot_labels_list(self.dataframe[self.target_column][index]))


MAX_LEN = 252
HIDDEN_SIZE = 768
NUM_EPOCHS = 10
PRINT_LOSS_EVERY = 10
train_dataframe = pd.read_csv("/Users/alexiskaldany/school/nlp/TEST_2/covid.csv").rename(columns={"content": "input", "category": "target"})
labels = train_dataframe['target'].unique()
NUM_OF_CLASSES = len(labels)
train_dataset = CustomDataset(train_dataframe, input_column = "input", target_column = "target", max_len = MAX_LEN)
print(train_dataset[0][0].shape)
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = MLP(MAX_LEN, MAX_LEN, NUM_OF_CLASSES).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

""" Train the MLP"""

for epoch in range(NUM_EPOCHS):
    for index, batch in enumerate(train_dataset):
        input = torch.tensor(batch[0][0]).float().to(device)
        target = batch[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if index % PRINT_LOSS_EVERY == 0:
            print(f"Epoch: {epoch} Loss: {loss.item()}")

"""
RNN
"""

""" 
Transformer
"""
model1_name = 'howey/electra-small-mnli'
