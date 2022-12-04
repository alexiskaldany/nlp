import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

"""Bert as base model with LSTM on top """
class bertLSTM(nn.Module):
    
    def __init__(self) -> None:
        super(bertLSTM, self).__init__()
        self.base_model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2",ignore_mismatched_sizes=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        lstm_output, _ = self.lstm(base_output.last_hidden_state)
        logits = self.classifier(lstm_output[:,0,:])
        return logits

# import torch.nn as nn
# from transformers import AutoModel
# model = AutoModel.from_pretrained("bert-base-uncased",ignore_mismatched_sizes=True)
# print(model)
# model.add_module("lstm",nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True))
# print(model)
# model.classifier = 
# # print(model)

# """ Bert as base model with LSTM on top, with classifier on top of LSTM, for multi-label classification"""
# model = AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2",ignore_mismatched_sizes=True)
model = bertLSTM()
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
print(tokenizer)
# model.add_module("classifier",nn.Linear(128, 2))

print(model)
import torch
input = tokenizer("This is a test string", return_tensors="pt")
print(input)
test_label = torch.tensor([1,0])
output = model(**input)
print(output)

argmax = torch.argmax(output, dim=1)


