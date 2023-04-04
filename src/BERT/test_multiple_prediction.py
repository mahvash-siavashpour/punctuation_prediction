import torch
import argparse
import json
import sys
import numpy as np
import pandas as pd
# import language_lists
from transformers import AutoTokenizer


from mylib import bert_train_func
from mylib import config, dataload_func


def read_data(file_name, nrows, chunksize, seq_shift):
  df = pd.read_csv(file_name,sep=',', nrows=nrows)
  text = df.iloc[:, 0].values
  tags = df.iloc[:, 1].values
  input_tokens = []
  input_labels = []
  i = 0
  while i< len(tags):

    input_tokens.append(text[i:i+chunksize])
    input_labels.append(tags[i:i+chunksize])
  
    i += seq_shift


  return input_tokens, input_labels



parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')

parser.add_argument('seq_shift',
                    help='Sequence Shift')
args = parser.parse_args()
seq_shift = args.seq_shift

### read models configuration json file
with open("bert_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())


configurations = config.SetModelConfig(args.model_name, models)

sys.stdout = open(configurations["log_file_path_inference"], 'w', encoding="utf-8")


unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]


bert_model_name = configurations["bert_model_name"]
chunksize = configurations["chunksize"]
loss_fct = bert_train_func.loss_fct(weights=None)


model = bert_train_func.CustomModel(num_classes=len(list(unique_tags)), loss_fct=loss_fct, bert_model_name=bert_model_name, model_type=configurations["model_architecture"])


model.load_state_dict(torch.load(configurations["save_model_path"]))
model.eval()

print(model)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
test_tokens, test_tags = read_data(configurations["test_file_name"], configurations["test_data_size"], chunksize=configurations["chunksize"], seq_shift=seq_shift)

testing_set = dataload_func.MyDataset(text=test_tokens, tags=test_tags, tokenizer=tokenizer, tag2id=tag2id, max_len=configurations["bert_seq_max_len"])

test_params = {'batch_size': 64,
                'shuffle': False,
                'num_workers': 2
                }

testing_loader = torch.utils.data.DataLoader(testing_set, **test_params)


all_valid_preds = []
for data in testing_loader:
    text, targets = data
    with torch.no_grad():
        outputs = model(text)

    outputs=outputs['logits']

    predictions = outputs.detach().cpu().numpy()
    # predictions = np.argmax(predictions, axis=2)


    # all_valid_preds.append(predictions.detach().cpu().numpy())

print(predictions.shape)
# i = 0
# while i< predictions.shape[0]:
#    tmp_pred = []

#    if i ==0:
#       tmp_pred = predictions


# # Remove ignored index (special tokens)
# true_predictions = [
#     [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]
# true_labels = [
#     [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]

# print()
# results = metric.compute(predictions=true_predictions, references=true_labels)
# print(f"**Testing Set Results** \n {results}")

    
# pm = performance_measure(true_labels, true_predictions)
# print(f"\n{pm}\n")
# TP = pm['TP']+pm['TN']
# TN = pm['TN']
# FP=pm['FP']
# FN=pm['FN']
# f1 = TP/(TP+(.5*(FP+FN)))
# f1_O = TN/(TN+(.5*(FP+FN)))
# print(f"overall f1 with TN: {f1}")
# print(f"O f1: {f1_O}")


