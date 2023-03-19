import torch
import numpy as np
from mylib import cnnlstm_train_func
from mylib import dataload_func
from mylib import config
import argparse
import json
import sys


from datasets import load_metric
metric = load_metric("seqeval")

from mylib import bert_train_func
from mylib import config

parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')
args = parser.parse_args()


### read models configuration json file
with open("cnnlstm_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())



configurations = config.SetModelConfig(args.model_name, models)

sys.stdout = open(configurations["log_file_path_inference"], 'w', encoding="utf-8")


unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]


model = cnnlstm_train_func.CNN_LSTM(input_size=configurations['input_size'], hidden_size=configurations['lstm_hidden_size'], num_layers=1, num_classes=len(list(unique_tags)))


model.load_state_dict(torch.load(configurations["save_model_path"]))
model.eval()

print(model)


def get_punc(text, splitted=False):
  if not splitted:
    text = text.split()
  x_prepared = dataload_func.get_embedding(text)

  X = torch.from_numpy(x_prepared).float()
  X = X.reshape(1, 300, 10)
  out = model(X)
  out = out.detach().numpy()
  new_outputs = np.argmax(out,axis=2)

  result = []
  for o, t in zip(new_outputs[0], text):
    result.append((t, id2tag[o]))

  return result


text = "من در ایران زندگی میکنم ولی شما چطور زندگی میکنید"
print(get_punc(text))
