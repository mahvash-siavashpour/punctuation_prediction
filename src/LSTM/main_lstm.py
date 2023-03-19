# -*- coding: utf-8 -*-
"""12.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1pVjOY-DmBnHBXQHETGRj6wc_QhwiLm5l
"""

root = "../../Data/"
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import fasttext


from mylib import lstm_train_func
from mylib import dataload_func
from mylib import config

import argparse
import json
import sys

from datasets import load_metric
metric = load_metric("seqeval")

punc = ['.', '،', '؟', '!']



parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')
args = parser.parse_args()


### read models configuration json file
with open("lstm_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())




configurations = config.SetModelConfig(args.model_name, models)
sys.stdout = open(configurations["log_file_path"], 'w', encoding="utf-8")



unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]



fasttext_model = fasttext.load_model('word-embeddings/cc.fa.300.bin')
x_train, y_train = dataload_func.read_data(configurations["train_file_name"], configurations["train_data_size"], seq_size=configurations["lstm_seq_max_len"])
x_test, y_test = dataload_func.read_data(configurations["test_file_name"], configurations["test_data_size"], seq_size=configurations["lstm_seq_max_len"])



model = lstm_train_func.LSTM_Model(input_size=configurations['input_size'], hidden_size=configurations['lstm_hidden_size'], 
                                    num_layers=1, num_classes=len(list(unique_tags)), use_cnn=configurations['use_cnn'])

print(model)


from torch.utils.data import DataLoader



#Datasets
train_dataset = dataload_func.MyDataset(x_train, y_train, tag2id, fasttext_model)
test_dataset = dataload_func.MyDataset(x_test, y_test, tag2id, fasttext_model)
# Parameters

train_params = {'batch_size': 32,
                'shuffle': False,
                'num_workers': 0
                }

test_params = {'batch_size': 32,
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




model = lstm_train_func.train(epochs=EPOCHS, model=model, writer=writer, 
                                     train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                                     loss_function=loss_function, id2tag=id2tag, timestamp=timestamp, best_vloss=best_vloss,
                                     num_classes=len(list(unique_tags)))

# save model

torch.save(model.state_dict(), configurations["save_model_path"])



# import inference_lstm
# text = "من در ایران زندگی میکنم ولی شما چطور زندگی میکنید"
# print(inference_lstm.get_punc(text))


