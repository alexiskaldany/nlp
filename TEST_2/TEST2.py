import transformers
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
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
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device("cuda")
if not torch.cuda.is_available():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
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
BATCH_SIZE = 384
"""
Read in dataset here
"""
directory = Path(__file__).parent.absolute()
train = pd.read_csv(directory/"data_example/train.csv")
val = pd.read_csv(directory/"data_example/val.csv")
test = pd.read_csv(directory/"data_example/test.csv")
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
        encoding = self.tokenizer.encode_plus(
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
            "labels": label,
        }
train_seq_list = train[sequence_col].tolist()
train_label_list = train[label_col].tolist()
train_label_list = [torch.tensor(label, dtype=torch.float32) for label in train_label_list]
train_label_list = [torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
    torch.float
) for label in train_label_list]
val_seq_list = val[sequence_col].tolist()
val_label_list = val[label_col].tolist()
val_label_list = [torch.tensor(label, dtype=torch.float32) for label in val_label_list]
val_label_list = [torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES).to(
    torch.float
) for label in val_label_list]
test_seq_list = test[sequence_col].tolist()
test_label_list = test[label_col].tolist()
test_label_list = [torch.tensor(label, dtype=torch.float32).to(
    torch.float
) for label in test_label_list]
test_label_list = [torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.int64), NUM_CLASSES)for label in test_label_list]



train_dataset = Multiclass(sequence_list=train_seq_list, label_list=train_label_list, tokenizer=AutoTokenizer.from_pretrained(model1_name), max_len=MAX_LENGTH, num_of_classes=NUM_CLASSES)
val_dataset = Multiclass(sequence_list=val_seq_list, label_list=val_label_list, tokenizer=AutoTokenizer.from_pretrained(model1_name), max_len=MAX_LENGTH, num_of_classes=NUM_CLASSES)
test_dataset = Multiclass(sequence_list=test_seq_list, label_list=test_label_list, tokenizer=AutoTokenizer.from_pretrained(model1_name), max_len=MAX_LENGTH, num_of_classes=NUM_CLASSES)

## Metrics

def compute_metrics(eval_preds):
    metric = load_metric('f1','accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    results = metric.compute(predictions=predictions, references=labels_argmax, average='macro')
    print(results)
    return results


## Model 1
print("Model 1")
model_1_tokenizer = AutoTokenizer.from_pretrained(model1_name)
model_1 = AutoModelForSequenceClassification.from_pretrained(model1_name, num_labels=NUM_CLASSES,ignore_mismatched_sizes=True)
model_1_collator = DataCollatorWithPadding(tokenizer=model_1_tokenizer)

training_args = TrainingArguments(
    output_dir="./results_binary",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    num_train_epochs=2,
    weight_decay=0.01,
    seed=seed,
)
model1_trainer = Trainer(
    model=model_1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=model_1_tokenizer,
    data_collator=model_1_collator,
    compute_metrics=compute_metrics,
)
print("Training Model 1")
model1_trainer.train()
model1_trainer.save_model(directory / 'results_binary/' + 'model_1')
model1_results = model1_trainer.evaluate()

with open(directory / 'results_binary/model_1/model1_results.txt', 'w') as f:
    print(model1_results, file=f)

del model1_trainer
gc.collect()
torch.cuda.empty_cache()
    
    
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

model1_tokenizer = AutoTokenizer.from_pretrained(model1_name)
model1 = AutoModelForSequenceClassification.from_pretrained(model1_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
print(f'\n\n{model1_name} number of parameters:', model1.num_parameters())
