import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import random
from datasets import load_metric, load_dataset, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import gc
import warnings
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
if not torch.cuda.is_available():
    device = torch.device("cpu")
    if torch.has_mps:
        device = torch.device("mps")
print(device)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
MAX_LENGTH = 128
LR = 2e-5
EPOCHS = 3
"""Load data"""
current_dir = Path(__file__).parent
print(current_dir)
TRAIN_PATH = current_dir / "data" / "train.csv"
TEST_PATH = current_dir / "data" / "test.csv"
# OUTPUT_PATH = current_dir / "output"
MLP_PATH = current_dir / "mlp"
LSTM_PATH = current_dir / "lstm"
CNN_PATH = current_dir / "cnn"
train_df = pd.read_csv(str(TRAIN_PATH),nrows=500).rename(
    columns={"content": "text", "category": "label"}
)
NUM_CLASSES = train_df["label"].nunique()
LABEL_TO_ID = {label: int(i) for i, label in enumerate(train_df["label"].unique())}
test_df = pd.read_csv(str(TEST_PATH),nrows=500).rename(
    columns={"content": "text", "category": "label"}
)
print(train_df.head())


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
        label = self.label_list[index]  # TODO labels are tricky
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
train_output = train_df["label"].tolist()
train_label_list = [LABEL_TO_ID[label] for label in train_output]
train_label_list = [
    torch.tensor(label, dtype=torch.float32) for label in train_label_list
]
train_label_list = [
    torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
        torch.float
    )
    for label in train_label_list
]

test_input = test_df["text"].tolist()
test_output = test_df["label"].tolist()
test_label_list = [LABEL_TO_ID[label] for label in test_output]
test_label_list = [
    torch.tensor(label, dtype=torch.float32) for label in test_label_list
]
test_label_list = [
    torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
        torch.float
    )
    for label in test_label_list
]

tokenizer = AutoTokenizer.from_pretrained(
    "cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True
)
train_ds = Bert_DS(
    sequence_list=train_input,
    label_list=train_label_list,
    tokenizer=tokenizer,
    max_len=MAX_LENGTH,
)
test_ds = Bert_DS(
    sequence_list=test_input,
    label_list=test_label_list,
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


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH
)
""" Models"""
from torch import nn


class bertMLP(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(bertMLP, self).__init__()
        self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

bertMLP = bertMLP(num_classes=NUM_CLASSES)

class bertLSTM(nn.Module):
    def __init__(self, num_classes):
        super(bertLSTM, self).__init__()
        self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        lstm_out, (h_n, c_n) = self.lstm(last_hidden_states)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits

bertLSTM = bertLSTM(num_classes=NUM_CLASSES)

class bertCNN(nn.Module):
    def __init__(self, num_classes):
        super(bertCNN, self).__init__()
        self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2", ignore_mismatched_sizes=True)
        self.conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        conv_out = self.conv(last_hidden_states.transpose(1, 2))
        logits = self.classifier(conv_out[:, -1, :])
        return logits

bertCNN = bertCNN(num_classes=NUM_CLASSES)

from torch.optim import AdamW
optimizerMLP = AdamW(bertMLP.parameters(), lr=1e-5)
# MLP


for epoch in range(EPOCHS):
    for batch in range(len(train_ds)):
        input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
        attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
        labels = train_ds[batch]["labels"].unsqueeze(0)
        outputs = bertMLP(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        optimizerMLP.step()
        optimizerMLP.zero_grad()
        # predicted = torch.argmax(outputs, dim=1)
        # labels = torch.argmax(labels, dim=1)
        # print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", loss.item(), "Predicted: ", predicted.item(), "Labels: ", labels.item())
# testing model using accuracy score as metric
accuracy = []
for batch in range(len(test_ds)):
    input_ids = test_ds[batch]["input_ids"].unsqueeze(0)
    attention_mask = test_ds[batch]["attention_mask"].unsqueeze(0)
    labels = test_ds[batch]["labels"].unsqueeze(0)
    outputs = bertMLP(input_ids=input_ids, attention_mask=attention_mask)
    labels = torch.argmax(labels, dim=1)
    predicted = torch.argmax(outputs, dim=1)
    accuracy.append(predicted.item() == labels.item())
    
    

mlp_accuracy = sum(accuracy)/len(accuracy)
print("MLP Accuracy: ", mlp_accuracy)
accuracy_all_models = pd.DataFrame(columns=["Model", "Accuracy"])
accuracy_all_models = accuracy_all_models.append({"Model": "MLP", "Accuracy": mlp_accuracy}, ignore_index=True)
accuracy_all_models.to_csv("accuracy_all_models.csv", index=False)

"""LSTM"""
optimizerLSTM = AdamW(bertLSTM.parameters(), lr=1e-5)

for epoch in range(EPOCHS):
    for batch in range(len(train_ds)):
        input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
        attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
        labels = train_ds[batch]["labels"].unsqueeze(0)
        outputs = bertLSTM(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        optimizerLSTM.step()
        optimizerLSTM.zero_grad()
        # predicted = torch.argmax(outputs, dim=1)
        # labels = torch.argmax(labels, dim=1)
        # print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", loss.item(), "Predicted: ", predicted.item(), "Labels: ", labels.item())
        
accuracy = []
for batch in range(len(test_ds)):
    input_ids = test_ds[batch]["input_ids"].unsqueeze(0)
    attention_mask = test_ds[batch]["attention_mask"].unsqueeze(0)
    labels = test_ds[batch]["labels"].unsqueeze(0)
    outputs = bertLSTM(input_ids=input_ids, attention_mask=attention_mask)
    labels = torch.argmax(labels, dim=1)
    predicted = torch.argmax(outputs, dim=1)
    accuracy.append(predicted.item() == labels.item())
    
lstm_accuracy = sum(accuracy)/len(accuracy)
accuracy_all_models = accuracy_all_models.append({"Model": "LSTM", "Accuracy": lstm_accuracy}, ignore_index=True)
accuracy_all_models.to_csv("accuracy_all_models.csv", index=False)

"""CNN"""

optimizerCNN = AdamW(bertCNN.parameters(), lr=1e-5)

for epoch in range(EPOCHS):
    for batch in range(len(train_ds)):
        input_ids = train_ds[batch]["input_ids"].unsqueeze(0)
        attention_mask = train_ds[batch]["attention_mask"].unsqueeze(0)
        labels = train_ds[batch]["labels"].unsqueeze(0)
        outputs = bertCNN(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        optimizerCNN.step()
        optimizerCNN.zero_grad()

accuracy = []

for batch in range(len(test_ds)):
    input_ids = test_ds[batch]["input_ids"].unsqueeze(0)
    attention_mask = test_ds[batch]["attention_mask"].unsqueeze(0)
    labels = test_ds[batch]["labels"].unsqueeze(0)
    outputs = bertCNN(input_ids=input_ids, attention_mask=attention_mask)
    labels = torch.argmax(labels, dim=1)
    predicted = torch.argmax(outputs, dim=1)
    accuracy.append(predicted.item() == labels.item())
    
cnn_accuracy = sum(accuracy)/len(accuracy)
accuracy_all_models = accuracy_all_models.append({"Model": "CNN", "Accuracy": cnn_accuracy}, ignore_index=True)
accuracy_all_models.to_csv("accuracy_all_models.csv", index=False)