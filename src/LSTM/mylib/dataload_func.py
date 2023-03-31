import torch
import pandas as pd
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import fasttext



def read_data(file_name, nrows, seq_size):
  df = pd.read_csv(file_name,sep=',', nrows=nrows)
  text = df.iloc[:, 0].values
  tags = df.iloc[:, 1].values
  i = 0 
  input_tokens = []
  input_labels = []

  tmp_tokens = []
  tmp_tags = []

  for word, label in zip(text, tags):
    if i % seq_size == 0 and i !=0:
      input_tokens.append(tmp_tokens)
      input_labels.append(tmp_tags)
      tmp_tokens = []
      tmp_tags = []

    tmp_tokens.append(word)
    tmp_tags.append(label)
    i += 1

  if len(input_tokens) == 0 and len(tmp_tokens) !=0:
    input_tokens = tmp_tokens
    input_labels = tmp_tags

  return input_tokens, input_labels



def get_embedding(words_list, fasttext_model, EMBEDDING_DIM=300):

  X = []
  for word in words_list:    
    X.append(fasttext_model[word])
  X = np.array(X)
  return X



def tag_to_id(tags_list, tag2id):
    # print(tags_list)
    y = [tag2id[s] for s in tags_list]
    # print(y)
    y = np.array(y)
    return y


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,text, tags, tag2id, fasttext_model):
      
        self.embedding = get_embedding
        self.tag_to_id = tag_to_id
        self.text = text
        self.tags = tags
        self.tag2id = tag2id
        self.fasttext_model = fasttext_model

    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):
        x_data = self.text[index]
        y_data = self.tags[index]
        # print(x_data, y_data)
        x_prepared = self.embedding(x_data, self.fasttext_model)
        y_prepared = self.tag_to_id(y_data, self.tag2id)
        # print(x_prepared.shape)
        X = torch.from_numpy(x_prepared).float()
        Y = torch.from_numpy(y_prepared).long()
        item = {"x": X, "y": Y}
        
        return item


def label_counts(id2tag, train_loader):
  label_count = {}
  total_labels = 0
  for u in id2tag.keys():
    label_count[str(u)] = 0

  for idx, batch in enumerate(train_loader):
    labels = batch['y']

    for data in labels:
      for e in data:

        if e == -100:
          continue
        label_count[str(int(e))] += 1
        total_labels += 1

  return label_count, total_labels



def loss_weights(label_count, total_labels):
    weights = []
    for i in label_count.keys():
        # print(f"{label_frq[i]/total_labels},   {np.log(label_frq[i])/np.log(total_labels)}")
        # print()
        # w = 1 - total_labels/label_count[i]
        w =  np.sqrt(total_labels/label_count[i])
        # w = np.sqrt(1/label_count[i])
        # b = 0.9
        # w = 1/((1- b**np.log(label_count[i]))/(1-b))
        weights.append(w)
        # print(w)

    weights = np.array(weights)
    weights = weights / np.sum(weights)
        
    return weights

