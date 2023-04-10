import torch
import argparse
import json
import sys
import numpy as np
import pandas as pd
# import language_lists
from transformers import AutoTokenizer
from seqeval.metrics import performance_measure
from datasets import load_metric
metric = load_metric("seqeval")

from mylib import bert_train_func
from mylib import config, dataload_func
import time

# python3 test_multiple_prediction.py bert_2.7 --list 2 4 8 16 >> ../../logs/Inference/2.txt

def read_data(file_name, nrows, chunksize, seq_shift):
  df = pd.read_csv(file_name,sep=',', nrows=nrows)
  text = df.iloc[:, 0].values
  tags = df.iloc[:, 1].values
  input_tokens = []
  input_labels = []
  i = 0
  print(len(tags))
  while i+chunksize <= len(tags):

    input_tokens.append(list(text[i:i+chunksize]))
    input_labels.append(list(tags[i:i+chunksize]))
  
    i += seq_shift


  return input_tokens, input_labels



def combine_predictions(pred_num, good_label, good_pred, seq_shift):

    ps = []
    final_labels = []
    for i in range(good_pred.shape[0]):#+512//pred_num-1):

        start_idx = max(0, i-pred_num+1)
        end_idx = min(good_pred.shape[0], i+1)
        print(f"**{start_idx, end_idx}")
        p = []
        flag = False
        for j, k in enumerate(range(start_idx, end_idx)):
            j = end_idx - start_idx - j - 1

            print(k, j)
            p.append(good_pred[k][j*seq_shift:(j+1)*seq_shift])
            if not flag:
              final_labels.append(good_label[k][j*seq_shift:(j+1)*seq_shift])
              flag = True

        if p is not []:
          print(len(p))
          print(p[0].shape)
          if p[0].shape[0] == 0:
            print(p)
          ps.append(np.log(np.exp(p).mean(0)))
    
    for i in range(good_pred.shape[0], good_pred.shape[0]+pred_num):
        start_idx = max(0, i-pred_num+1)
        end_idx = min(good_pred.shape[0], i+1)
        if start_idx == end_idx:
          break
        print(f"**{start_idx, end_idx}")
        p = []
        flag = False
        for j, k in enumerate(range(start_idx, end_idx)):
            j = pred_num-1 - j
            print(k, j)
            p.append(good_pred[k][j*seq_shift:(j+1)*seq_shift])
            if not flag:
              final_labels.append(good_label[k][j*seq_shift:(j+1)*seq_shift])
              flag = True

        if p is not []:
          ps.append(np.log(np.exp(p).mean(0)))

    # ps = np.array(ps)
    ps = np.concatenate(ps)
    print("**")
    print(len(final_labels))
    final_labels = np.concatenate(final_labels)
    
    return ps, final_labels


parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')

parser.add_argument('--list', type=str, nargs='+', default=[])

args = parser.parse_args()


if args.list == []:
    print("Enter one arg at least")

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


pred_num = args.list


model = bert_train_func.CustomModel(num_classes=len(list(unique_tags)), loss_fct=loss_fct, bert_model_name=bert_model_name, model_type=configurations["model_architecture"])


model.load_state_dict(torch.load(configurations["save_model_path"]))
model = model.to(device)
model.eval()

print(model)

tokenizer = AutoTokenizer.from_pretrained(bert_model_name)




"""
---------MAIN------------
"""


st = time.time()

device = torch.device("cuda:0")


"""
Loop over prediction numbers
"""

for pn in pred_num:

  pn = int(pn)
  seq_shift =  configurations["chunksize"] // pn
  test_size = (configurations["test_data_size"] // configurations["chunksize"])*configurations["chunksize"]


  test_tokens, test_tags = read_data(configurations["test_file_name"], test_size, chunksize=configurations["chunksize"], seq_shift=seq_shift)
  testing_set = dataload_func.MyDataset(text=test_tokens, tags=test_tags, tokenizer=tokenizer, tag2id=tag2id, max_len=configurations["bert_seq_max_len"])

  test_params = {'batch_size': 64,
                  'shuffle': False,
                  'num_workers': 2
                  }

  testing_loader = torch.utils.data.DataLoader(testing_set, **test_params)

  


  all_valid_preds = []
  all_valid_labels = []
  for idx, batch in enumerate(testing_loader):
      labels = batch['labels'].to(device)
      text = batch['input_ids'].to(device)

      with torch.no_grad():
          outputs = model(text)

      outputs=outputs['logits']

      predictions = outputs.cpu().detach().numpy()
      # predictions = np.argmax(predictions, axis=2)


      all_valid_preds.append(predictions)
      all_valid_labels.append(np.array(labels))

  all_valid_preds = np.concatenate(all_valid_preds)
  print(all_valid_preds.shape)

  all_valid_labels = np.concatenate(all_valid_labels)
  print(all_valid_labels.shape)



  # Remove ignored index (special tokens)

  good_pred = []
  good_label = []
  for prediction, label in zip(all_valid_preds, all_valid_labels):
      tmp_pred = []
      tmp_label = []

      for (p, l) in zip(prediction, label):
        if l != -100:
          tmp_pred.append(p)
          tmp_label.append(l)

      good_pred.append(np.array(tmp_pred))
      good_label.append(np.array(tmp_label))


  good_label = np.array(good_label)
  good_pred = np.array(good_pred)




  ps, fl = combine_predictions(pn, good_label, good_pred, seq_shift)

  final_predictions = np.argmax(ps, axis=1)
  print(final_predictions.shape)


  true_predictions = []
  true_labels = []
  for p, l in zip(final_predictions, fl):
      true_predictions.append(id2tag[p])
      true_labels.append(id2tag[l])


  size = 64

  final_true_predictions = []
  final_true_labels = []
  for i in range(len(true_labels)//size):
    final_true_predictions.append(true_predictions[i:i+size])
    final_true_labels.append(true_labels[i:i+size])


  results = metric.compute(predictions=final_true_predictions, references=final_true_labels)



  et = time.time()

  # get the execution time
  elapsed_time = et - st
  print('Execution time:', elapsed_time, 'seconds')



  print(f"Number of Predictions >> {pn}")
  print(results)


  pm = performance_measure(true_labels, true_predictions)
  print(f"\n{pm}\n")
  TP = pm['TP']+pm['TN']
  TN = pm['TN']
  FP=pm['FP']
  FN=pm['FN']
  f1 = TP/(TP+(.5*(FP+FN)))
  f1_O = TN/(TN+(.5*(FP+FN)))
  print(f1)
  print(f1_O)

