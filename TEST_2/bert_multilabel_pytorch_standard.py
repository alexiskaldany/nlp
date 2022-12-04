# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
import sys


cur_dir = 
train_path =
test_path = "/content/drive/MyDrive/Colab Notebooks/HuggingFace/dataset/test.csv"

import os
os.getcwd()

train_df = pd.read_csv(train_path,nrows = 2000)
test_df = pd.read_csv(test_path)

train_df.columns

test_df.head()

# combining 'title' and 'abstract' column to| get more context
train_df['CONTEXT'] = train_df['TITLE'] + ". " + train_df['ABSTRACT']
test_df['CONTEXT'] = test_df['TITLE'] + ". " + test_df['ABSTRACT']

# dropping useless features/columns
train_df.drop(labels=['TITLE', 'ABSTRACT', 'ID'], axis=1, inplace=True)
test_df.drop(labels=['TITLE', 'ABSTRACT', 'ID'], axis=1, inplace=True)

train_df.columns

# rearranging columns
train_df = train_df[['CONTEXT', 'Computer Science', 'Physics', 'Mathematics', 'Statistics',
                     'Quantitative Biology', 'Quantitative Finance',]]

train_df.head()

target_list = ['Computer Science', 'Physics', 'Mathematics', 'Statistics',
       'Quantitative Biology', 'Quantitative Finance']

# hyperparameters
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-05

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

example_text = 'do like and subcribe channel'
encondings = tokenizer.encode_plus(
    example_text,
    add_special_tokens = True,
    max_length = MAX_LEN,
    padding = 'max_length',
    truncation = True,
    return_attention_mask = True,
    return_tensors = 'pt'                               )

encondings

"""## Loading/Tokenizing Data in Pytorch"""

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['CONTEXT']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }

train_size = 0.8
train_df2 = train_df.sample(frac=train_size, random_state=200)
val_df = train_df.drop(train_df2.index).reset_index(drop=True)
train_df=train_df2.reset_index(drop=True)

train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device

!pip install GPUtil
from GPUtil import showUtilization as gpu_usage
gpu_usage()

# import torch
# torch.cuda.empty_cache()
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)

"""## Selecting Model """

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        #self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(768, 6) #change number of labels
        #self.relu = nn.ReLU()
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )       
        output_dropout = self.dropout(output.pooler_output)
        #output, hidden = self.lstm(output_dropout)
        #linear_outout = self.linear(output)
        #final_layer = self.relu(linear_output)
        output = self.linear(output_dropout) #comment out if you only use lstm
        return output #final_layer for lstm

class MLPClass(torch.nn.Module): #with MLP
    def __init__(self):
        super(MLPClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6) 
                ### 1st hidden layer
        self.linear_1 = torch.nn.Linear(768,100)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
                ### Output layer
        self.linear_out = torch.nn.Linear(100, 6) #change number of labels
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )       
        output_dropout = self.dropout(output.pooler_output)
        out = self.linear_1(output_dropout)
        out = torch.sigmoid(out)
        logits = self.linear_out(out)
        return logits

class CNNClass(torch.nn.Module):
    def __init__(self):
        super(CNNClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.Conv1 = nn.Conv1d(10,10,1) #this shit doesnt not work 
        self.linear = torch.nn.Linear(768, 6) #change number of labels
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )       
        output_dropout = self.dropout(output.pooler_output)
        output = self.Conv1(output_dropout)
        output = self.linear(output)
        return output

model = BERTClass()

model.add_module("lstm",nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)) # it works without using linear after the lstm ... best clean way to implement LSTM

model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

val_targets=[]
val_outputs=[]

def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf
  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        #print('yyy epoch', batch_idx)
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        #if batch_idx%5000==0:
         #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader) 
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))    
    print('############# Epoch {}  Done   #############\n'.format(epoch))
  return model

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer)

"""## Testing Model"""

testing = test_df['CONTEXT'].tolist()[0:100]

len(testing)

def return_whole_vals(x):
  fin = []
  for i in x:
    if i > 0.50: # select your threshold
      fin.append(1)
    else:
      fin.append(0)
  return fin

# testing
y_predict = []
for example in testing:
  #example = test_df['CONTEXT'][0]
  encodings = tokenizer.encode_plus(
      example,
      None,
      add_special_tokens=True,
      max_length=MAX_LEN,
      padding='max_length',
      return_token_type_ids=True,
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
  )
  trained_model.eval()
  with torch.no_grad():
      input_ids = encodings['input_ids'].to(device, dtype=torch.long)
      attention_mask = encodings['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = encodings['token_type_ids'].to(device, dtype=torch.long)
      output = trained_model(input_ids, attention_mask, token_type_ids) 
      final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()
      y_predict.append([return_whole_vals(x) for x in final_output])

test = [x for listy in y_predict for x in listy]

sub_df = pd.DataFrame(test, columns=['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance'])

sub_df.sample(10) #if test has ID put here

