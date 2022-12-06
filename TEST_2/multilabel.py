""" 

Final Exam 2
NLP
Fall 2022
auther: Alexis Kaldany

Comments: 

MLP and Convulutional Neural Network give inferior results to LSTM.
They are commented out for convenience.
I ran two epochs on the BertLSTM model, then ran the testing set through the model and generated the outputs 
Other than changing file paths, no other changes should be needed to run this code.

Model:

I use a very small version of the TinyBERT model, which is a BERT model that has been trained on the MS MARCO dataset.
I added an LSTM layer to the model, and then a linear layer to the output of the LSTM layer.
This generates a vector of length 6, which is the number of labels.
I use cross-entropy as my loss function to train the model.

For the results:

I used the same model to generate the results for the test set as I did for the training set.
The outputs are converted into a list
If the value is => 0.5, the label is assigned a 1, otherwise it is assigned a 0.

The results are then written into the submission fil;e

""" 


import transformers
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    
)
import random
from datasets import  Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
if torch.cuda.is_available():
    device = torch.device("cuda")
if not torch.cuda.is_available():
    device = torch.device("cpu")
    if torch.has_mps:
        device = torch.device("mps")
print(device)
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
MAX_LENGTH = 128
LR = 2e-5
EPOCHS = 2
"""Load data"""
current_dir = Path(__file__).parent
print(current_dir)
TRAIN_PATH = current_dir / "data" / "Train_ML.csv"
TEST_PATH = current_dir / "data" / "Test_submission_netid.csv"
# OUTPUT_PATH = current_dir / "output"
print(TRAIN_PATH)
print(TEST_PATH)
MLP_PATH = current_dir / "mlp"
LSTM_PATH = current_dir / "lstm"
CNN_PATH = current_dir / "cnn"
train_df = pd.read_csv(str(TRAIN_PATH)).rename(columns={"ABSTRACT": "text"})
test_df = pd.read_csv(TEST_PATH).rename(columns={"ABSTRACT": "text"})
label_columns = train_df.columns[4:]
print(label_columns)
output_path = str(current_dir / "2e5ce_all_models.csv")
print(output_path)
row_labels = []
for row in train_df[label_columns].values:
    row_labels.append(row)
value_counts = train_df[label_columns].sum(axis=0).sort_values(ascending=False)
print(value_counts)
labels_tensor = [torch.tensor(label, dtype=torch.float32) for label in row_labels]
print(labels_tensor[0])
NUM_CLASSES = len(label_columns)
print(NUM_CLASSES)
class Bert_DS(Dataset):
    def __init__(
        self, sequence_list: List[str], label_list: List[int], tokenizer, max_len: int
    ):
        self.sequence_list = sequence_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, index):
        sequence = self.sequence_list[index]
        label = self.label_list[index]
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label.flatten(),
        }

train_input = train_df["text"].tolist()
tokenizer = AutoTokenizer.from_pretrained(
    "cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True
)
train_ds = Bert_DS(
    sequence_list=train_input,
    label_list=labels_tensor,
    tokenizer=tokenizer,
    max_len=MAX_LENGTH,
)
test_ds = Bert_DS(
    sequence_list=test_df["text"].tolist(),
    label_list=[torch.tensor([0] * NUM_CLASSES, dtype=torch.float32)] * len(test_df),
    tokenizer=tokenizer,
    max_len=MAX_LENGTH,
)

print(train_ds[0])
print(test_ds[0])


################### Changes between multi
# def compute_metrics(eval_preds):
#     metric = load_metric("accuracy")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     labels_argmax = np.argmax(labels, axis=-1)
#     results = metric.compute(predictions=predictions, references=labels_argmax)
#     return results

""" Models"""
from torch import nn


# class bertMLP(nn.Module):
#     def __init__(self, num_classes, dropout=0.1):
#         super(bertMLP, self).__init__()
#         self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(MAX_LENGTH, num_classes)
#         )
#     def forward(self, input_ids, attention_mask):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]
#         logits = self.classifier(pooled_output)
#         return logits

# bertMLP = bertMLP(num_classes=NUM_CLASSES)
# bertMLP.to(device)
class bertLSTM(nn.Module):
    def __init__(self, num_classes):
        super(bertLSTM, self).__init__()
        self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
        self.lstm = nn.LSTM(input_size=MAX_LENGTH, hidden_size=MAX_LENGTH, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(MAX_LENGTH, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        lstm_out, (h_n, c_n) = self.lstm(last_hidden_states)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits

bertLSTM = bertLSTM(num_classes=NUM_CLASSES)
# bertLSTM = bertLSTM.to(device)
# class bertCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(bertCNN, self).__init__()
#         self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
#         self.conv = nn.Conv1d(in_channels=MAX_LENGTH, out_channels=MAX_LENGTH, kernel_size=3, stride=1, padding=1)
#         self.classifier = nn.Linear(128, num_classes)
    
#     def forward(self, input_ids, attention_mask):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_states = outputs[0]
#         conv_out = self.conv(last_hidden_states.transpose(1, 2))
#         logits = self.classifier(conv_out[:, -1, :])
#         return logits

# bertCNN = bertCNN(num_classes=NUM_CLASSES)
# # bertCNN.to(device)


# optimizerMLP = AdamW(bertMLP.parameters(), lr=LR)
# # MLP


# for epoch in tqdm(range(EPOCHS)):
#     for batch in tqdm(range(len(train_ds))):
#         input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
#         attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
#         labels = train_ds[batch]["labels"].unsqueeze(0)
#         outputs = bertMLP(input_ids=input_ids, attention_mask=attention_mask)
#         loss = torch.nn.functional.cross_entropy(outputs,labels)
#         loss.backward()
#         optimizerMLP.step()
#         optimizerMLP.zero_grad()
#         # predicted = torch.argmax(outputs, dim=1)
#         # labels = torch.argmax(labels, dim=1)
#         # print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", loss.item(), "Predicted: ", predicted.item(), "Labels: ", labels.item())
# # testing model using accuracy score as metric
# cross_entropy = []
# for batch in tqdm(range(len(train_ds))):
#     input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
#     attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
#     labels = train_ds[batch]["labels"].unsqueeze(0)
#     outputs = bertMLP(input_ids=input_ids, attention_mask=attention_mask)
#     cross_entropy.append(torch.nn.functional.cross_entropy(outputs,labels).item())
    
    

# mlp_ce = np.mean(cross_entropy)
# print("MLP CE: ", mlp_ce)
# ce_all_models = pd.DataFrame(columns=["Model", "CE"])
# ce_all_models = ce_all_models.append({"Model": "MLP", "CE": mlp_ce}, ignore_index=True)
# ce_all_models.to_csv(output_path, index=False)

"""LSTM"""
optimizerLSTM = AdamW(bertLSTM.parameters(), lr=LR)

for epoch in tqdm(range(EPOCHS)):
    for batch in tqdm(range(len(train_ds))):
        input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
        attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
        labels = train_ds[batch]["labels"].unsqueeze(0)
        outputs = bertLSTM(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        optimizerLSTM.step()
        optimizerLSTM.zero_grad()        

converted_output_final = []
""" converting output into 0 or 1"""
def convert_output_to_label(output):
    new_output = []
    output = output.detach().numpy().tolist()[0]
    for i in range(len(output)):
        if output[i] >= 0.5:
            new_output.append(1)
        else:
            new_output.append(0)
    return new_output
            
for batch in tqdm(range(len(test_ds))):
    input_ids = test_ds[batch]["input_ids"].unsqueeze(0)
    attention_mask = test_ds[batch]["attention_mask"].unsqueeze(0)
    outputs = bertLSTM(input_ids=input_ids, attention_mask=attention_mask)
    converted_output = convert_output_to_label(outputs)
    converted_output_final.append(converted_output)
    
test_df[label_columns[0]] = [converted_output_final[i][0] for i in range(len(converted_output_final))]
test_df[label_columns[1]] = [converted_output_final[i][1] for i in range(len(converted_output_final))]
test_df[label_columns[2]] = [converted_output_final[i][2] for i in range(len(converted_output_final))]
test_df[label_columns[3]] = [converted_output_final[i][3] for i in range(len(converted_output_final))]
test_df[label_columns[4]] = [converted_output_final[i][4] for i in range(len(converted_output_final))]
test_df[label_columns[5]] = [converted_output_final[i][5] for i in range(len(converted_output_final))]

test_df.to_csv("/Users/alexiskaldany/school/nlp/TEST_2/data/Test_submission_netid_test.csv", index=False)


    
    
# lstm_ce = np.mean(lstm_ce)
# print("LSTM CE: ", lstm_ce)
# ce_all_models = ce_all_models.append({"Model": "LSTM", "CE": lstm_ce}, ignore_index=True)
# ce_all_models.to_csv(output_path, index=False)

# """CNN"""

# optimizerCNN = AdamW(bertCNN.parameters(), lr=LR)

# for epoch in tqdm(range(EPOCHS)):
#     for batch in tqdm(range(len(train_ds))):
#         input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
#         attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
#         labels = train_ds[batch]["labels"].unsqueeze(0)
#         outputs = bertCNN(input_ids=input_ids, attention_mask=attention_mask)
#         loss = torch.nn.functional.cross_entropy(outputs,labels)
#         loss.backward()
#         optimizerCNN.step()
#         optimizerCNN.zero_grad()

# cnn_ce = []

# for batch in tqdm(range(len(train_ds))):
#     input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
#     attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
#     labels = train_ds[batch]["labels"].unsqueeze(0)
#     outputs = bertCNN(input_ids=input_ids, attention_mask=attention_mask)
#     cnn_ce.append(torch.nn.functional.cross_entropy(outputs,labels).item())
   
    
# cnn_ce = np.mean(cnn_ce)
# print("CNN CE: ", cnn_ce)
# ce_all_models = ce_all_models.append({"Model": "CNN", "CE": cnn_ce}, ignore_index=True)
# ce_all_models.to_csv(output_path, index=False)