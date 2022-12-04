import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
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
MAX_LENGTH = 512
LR = 1e-5
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
train_df = pd.read_csv(str(TRAIN_PATH)).rename(
    columns={"content": "text", "category": "label"}
)
NUM_CLASSES = train_df["label"].nunique()
LABEL_TO_ID = {label: int(i) for i, label in enumerate(train_df["label"].unique())}
test_df = pd.read_csv(str(TEST_PATH)).rename(
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
def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    results = metric.compute(predictions=predictions, references=labels_argmax)
    return results


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH
)

mlp_model = AutoModel.from_pretrained("bert-base-uncased")
print(mlp_model)
mlp_model.add_module("classifier", torch.nn.Linear(768, NUM_CLASSES))



training_args = TrainingArguments(
    output_dir=MLP_PATH,
    learning_rate=LR,
    # per_device_train_batch_size=BATCH_SIZE,
    # per_device_eval_batch_size=BATCH_SIZE,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=seed,
    evaluation_strategy="epoch",
)
mlp_trainer = Trainer(
    model=mlp_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
mlp_trainer.train()
mlp_trainer.evaluate()
"""LSTM"""

"""CNN"""
