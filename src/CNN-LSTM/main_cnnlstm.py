# -*- coding: utf-8 -*-
"""12.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1pVjOY-DmBnHBXQHETGRj6wc_QhwiLm5l
"""

root = "../../Data/"
import torch
import pandas as pd
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import fasttext


from mylib import cnnlstm_train_func
from mylib import dataload_func
from mylib import config
punc = ['.', '،', '؟', '!']
# raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
# train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
# test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'

import argparse
import json
import sys

from datasets import load_metric
metric = load_metric("seqeval")


parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')
args = parser.parse_args()


### read models configuration json file
with open("bert_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())




configurations = config.SetModelConfig(args.model_name, models)
sys.stdout = open(configurations["log_file_path"], 'w', encoding="utf-8")



unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]



fasttext_model = fasttext.load_model('word-embeddings/cc.fa.300.bin')
x_train, y_train = dataload_func.read_data(configurations["train_file_name"], configurations["train_data_size"], seq_size=configurations["cnnlstm_seq_max_len"])
x_test, y_test = dataload_func.read_data(configurations["test_file_name"], configurations["test_data_size"], seq_size=configurations["cnnlstm_seq_max_len"])



model = cnnlstm_train_func.CNN_LSTM(input_size=300, hidden_size=10, num_layers=1, num_classes=5)

print(model)


from torch.utils.data import DataLoader



#Datasets
train_dataset = dataload_func.MyDataset(x_train, y_train, tag2id, fasttext_model)
test_dataset = dataload_func.MyDataset(x_test, y_test, tag2id, fasttext_model)
# Parameters

train_params = {'batch_size': 8,
                'shuffle': False,
                'num_workers': 0
                }

test_params = {'batch_size': 8,
                'shuffle': False,
                'num_workers': 0
                }

#Dataloaders
train_loader = DataLoader(train_dataset,  **train_params)
test_loader = DataLoader(test_dataset, **test_params)



"""## Loss Scheme"""
label_count = dataload_func.label_counts(id2tag, train_loader)
print(f"number of each labels: \n {label_count}")
weights = dataload_func.loss_weights(label_count)
print(f"weigh of each label: {weights}")



from torch.optim import Adam
# from sklearn.metrics import f1_score   

optimizer = Adam(model.parameters(), lr=configurations["LEARNING_RATE"])
loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = configurations["EPOCHS"]
best_vloss = 1_000_000.




model = cnnlstm_train_func.train(epochs=EPOCHS, model=model, writer=writer, 
                                     train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                                     loss_function=loss_function, id2tag=id2tag, timestamp=timestamp, best_vloss=best_vloss)

# save model

torch.save(model.state_dict(), configurations["save_model_path"])



import inference_cnnlstm
text = "من در ایران زندگی میکنم ولی شما چطور زندگی میکنید"
print(inference_cnnlstm.get_punc(text))


