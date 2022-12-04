import transformers
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import random
from datasets import load_metric
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
import gc
import warnings
import json

warnings.filterwarnings("ignore")

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
# covid = pd.read_csv("/Users/alexiskaldany/school/nlp/TEST_2/covid_articles_raw.csv")[:1000]
# train = covid.sample(frac=0.8, random_state=0)
# train.to_csv("/Users/alexiskaldany/school/nlp/TEST_2/data_example/train.csv")
# val = covid.drop(train.index).sample(frac=0.5, random_state=0)
# val.to_csv("/Users/alexiskaldany/school/nlp/TEST_2/data_example/val.csv")
# test = covid.drop(train.index).drop(val.index)
# test.to_csv("/Users/alexiskaldany/school/nlp/TEST_2/data_example/test.csv")

model1_name = "howey/electra-small-mnli"
model2_name = "monologg/koelectra-small-finetuned-nsmc"
model3_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
model4_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
model5_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5
"""
Read in dataset here
"""
directory = Path(__file__).parent.absolute()
print(directory)
train = pd.read_csv(directory / "data_example/train.csv")
val = pd.read_csv(directory / "data_example/val.csv")
test = pd.read_csv(directory / "data_example/test.csv")
sequence_col = "content"
label_col = "category"
NUM_CLASSES = train[label_col].nunique()
LABEL_TO_ID = {label: int(i) for i, label in enumerate(train[label_col].unique())}
train[label_col] = train[label_col].map(LABEL_TO_ID)
val[label_col] = val[label_col].map(LABEL_TO_ID)
test[label_col] = test[label_col].map(LABEL_TO_ID)


""" 
Create a custom dataset class that inherits from torch.utils.data.Dataset
"""
print("Loading Data")


class CustomDataset(Dataset):
    def __init__(
        self, sequence_list: List[str], label_list: List[int], tokenizer, max_len: int
    ):
        self.sequence_list = sequence_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence_list)


class Multiclass(CustomDataset):
    def __init__(
        self,
        sequence_list: List[str],
        label_list: List[int],
        tokenizer,
        max_len: int,
        num_of_classes: int,
    ):
        super().__init__(sequence_list, label_list, tokenizer, max_len)
        self.num_of_classes = num_of_classes
        # self.encoded_labels = [
        #     torch.nn.functional.one_hot(
        #         torch.tensor(label, dtype=torch.float32), self.num_of_classes
        #     )
        #     for label in self.label_list
        # ]

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


train_seq_list = train[sequence_col].tolist()
train_label_list = train[label_col].tolist()
train_label_list = [
    torch.tensor(label, dtype=torch.float32) for label in train_label_list
]
train_label_list = [
    torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
        torch.float
    )
    for label in train_label_list
]

val_seq_list = val[sequence_col].tolist()
val_label_list = val[label_col].tolist()
val_label_list = [torch.tensor(label, dtype=torch.float32) for label in val_label_list]
val_label_list = [
    torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
        torch.float
    )
    for label in val_label_list
]

test_seq_list = test[sequence_col].tolist()
test_label_list = test[label_col].tolist()
test_label_list = [
    torch.tensor(label, dtype=torch.float32).to(torch.float)
    for label in test_label_list
]
test_label_list = [
    torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES)
    for label in test_label_list
]

train_dataset = Multiclass(
    sequence_list=train_seq_list,
    label_list=train_label_list,
    tokenizer=AutoTokenizer.from_pretrained(model1_name),
    max_len=MAX_LENGTH,
    num_of_classes=NUM_CLASSES,
)
val_dataset = Multiclass(
    sequence_list=val_seq_list,
    label_list=val_label_list,
    tokenizer=AutoTokenizer.from_pretrained(model1_name),
    max_len=MAX_LENGTH,
    num_of_classes=NUM_CLASSES,
)

test_dataset = Multiclass(
    sequence_list=test_seq_list,
    label_list=test_label_list,
    tokenizer=AutoTokenizer.from_pretrained(model1_name),
    max_len=MAX_LENGTH,
    num_of_classes=NUM_CLASSES,
)


## Metrics


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    results = metric.compute(predictions=predictions, references=labels_argmax)
    return results


## Model 1
print("Model 1")

model_1_tokenizer = AutoTokenizer.from_pretrained(model1_name, use_fast=True)
# model_1_config = AutoConfig.from_pretrained(model1_name, num_labels=NUM_CLASSES,ignore_mismatched_sizes=True, finetuning_task="multiclass")
model_1 = AutoModelForSequenceClassification.from_pretrained(
    model1_name,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)


model_1_collator = DataCollatorWithPadding(
    tokenizer=model_1_tokenizer, padding=True, max_length=MAX_LENGTH
)
model_1_dir = directory / "model_1"
training_args = TrainingArguments(
    output_dir=model_1_dir,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=seed,
    save_strategy="steps",
)
model_1_trainer = Trainer(
    model=model_1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=model_1_tokenizer,
    data_collator=model_1_collator,
    compute_metrics=compute_metrics,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
)
print("Training Model 1")
model_1_trainer.train()
model_1_trainer.save_model(model_1_dir / "saved_model")
model_1_results = model_1_trainer.evaluate()

with open(model_1_dir / "results.json", "w") as f:
    json.dump(model_1_results, f)

del (
    model_1_trainer,
    model_1,
    model_1_tokenizer,
    model_1_collator,
    model_1_dir,
    training_args,
    model_1_results,
)
gc.collect()
torch.cuda.empty_cache()
## Model 2
model_2_tokenizer = AutoTokenizer.from_pretrained(model2_name)
model_2 = AutoModelForSequenceClassification.from_pretrained(
    model2_name,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
model_2_collator = DataCollatorWithPadding(
    tokenizer=model_2_tokenizer, padding=True, max_length=MAX_LENGTH
)
model_2_dir = directory / "model_2"
training_args = TrainingArguments(
    output_dir=model_2_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    overwrite_output_dir=True,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    seed=seed,
)
model_2_trainer = Trainer(
    model=model_2,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=model_2_tokenizer,
    data_collator=model_2_collator,
    compute_metrics=compute_metrics,
)
model_2_trainer.train()
model_2_trainer.save_model(model_2_dir / "saved_model")
model_2_results = model_2_trainer.evaluate()

with open(model_2_dir / "results.json", "w") as f:
    json.dump(model_2_results, f)

del (
    model_2_trainer,
    model_2,
    model_2_tokenizer,
    model_2_collator,
    model_2_dir,
    training_args,
    model_2_results,
)

# class MultiLabel(CustomDataset):
#     def __init__(
#         self,
#         sequence_list: List[str],
#         label_list: List[List[int]],
#         tokenizer,
#         max_len: int,
#         num_of_classes: int,
#     ):
#         super().__init__(sequence_list, label_list, tokenizer, max_len)
#         self.num_of_classes = num_of_classes
#         self.encoding = self.tokenizer(
#             sequence_list,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         self.encoded_labels = [
#             torch.nn.functional.one_hot(
#                 torch.tensor(label, dtype=torch.float32), self.num_of_classes
#             )
#             for label in self.label_list
#         ]

#     def __getitem__(self, index):
#         sequence = self.sequence_list[index]
#         label = self.encoded_labels[index]  # TODO labels are tricky
#         encoding = self.tokenizer.encode_plus(
#             sequence,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         return {
#             "input_ids": encoding["input_ids"].flatten(),
#             "attention_mask": encoding["attention_mask"].flatten(),
#             "label": label,
#         }


""" 
Manipulate the head of the model according to the kind of task
"""

