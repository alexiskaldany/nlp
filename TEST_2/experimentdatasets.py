from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from pathlib import Path
import pandas as pd
data_example = Path(__file__).parent.absolute() / "data_example"
train_path = str(data_example / "train.csv")

val_path = str(data_example / "val.csv")

test_path = str(data_example / "test.csv")
train_df = pd.read_csv(train_path).drop(columns=["Unnamed: 0","title"]).rename(columns={"content":"text","category":"labels"})
val_df = pd.read_csv(val_path).drop(columns=["Unnamed: 0","title"]).rename(columns={"content":"text","category":"labels"})
test_df = pd.read_csv(test_path).drop(columns=["Unnamed: 0","title"]).rename(columns={"content":"text","category":"labels"})

NUM_CLASSES = len(train_df["labels"].unique())
MAX_LENGTH = 1024

dataset = load_dataset("csv", data_dir=str(data_example),data_files=["train.csv","val.csv","test.csv"]).rename_columns({"content":"text","category":"labels"}).remove_columns(['Unnamed: 0', 'title'])
dataset = dataset["train"].train_test_split(test_size=0.2)
def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["labels"], max_length=128, truncation=True, padding="max_length",return_tensors="pt")["input_ids"]
    print(labels)
    # model_inputs["labels"] = labels["input_ids"].squeeze()
    return model_inputs




model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print(dataset)


                      