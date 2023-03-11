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


punc = ['.', '،', '؟', '!']
# raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
# train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
# test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'


raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
train_file_name = 'Preprocessed/taaghche/'+'train_taaghche.csv'
test_file_name = 'Preprocessed/taaghche/'+'test_taaghche.csv'


fasttext_model = fasttext.load_model('word-embeddings/cc.fa.300.bin')

def read_data(file_name, nrows, seq_size=512):
  df = pd.read_csv(root+file_name,sep=',', nrows=nrows)
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


words, tags = read_data(train_file_name, 5000000, seq_size=10)

unique_tags = set({'_qMark', '_exMark', 'O', '_dot', '_comma'})
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


def get_embedding(words_list, EMBEDDING_DIM=300):

  X = []
  for word in words_list:    
    X.append(fasttext_model[word])
  X = np.array(X)
  return X



def tag_to_id(tags_list):
    # print(tags_list)
    y = [tag2id[s] for s in tags_list]
    # print(y)
    y = np.array(y)
    return y


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,text, tags, embedding=get_embedding, tag_to_id=tag_to_id):
      
        self.embedding = embedding
        self.tag_to_id = tag_to_id
        self.text = text
        self.tags = tags

    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):
        x_data = self.text[index]
        y_data = self.tags[index]
        # print(x_data, y_data)
        x_prepared = self.embedding(x_data)
        y_prepared = self.tag_to_id(y_data)
        # print(x_prepared.shape)
        X = torch.from_numpy(x_prepared).float()
        Y = torch.from_numpy(y_prepared).long()
        item = {"x": X, "y": Y}
        
        return item


#split into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(words, tags, test_size=0.2, random_state=1)



class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        #cnn takes input of shape (batch_size, channels, seq_len)
        # print(x.shape)
        
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        # print(out.shape)
        out, _ = self.lstm(out)

        # print(out.shape)

        predictions  = self.fc(out)
        # tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        # print(predictions.shape)
        return predictions 


model = CNN_LSTM(input_size=300, hidden_size=10, num_layers=1, num_classes=5)

print(model)


from torch.utils.data import DataLoader



#Datasets
train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)
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
  
print(label_count)


weights = []
for i in label_count.keys():
  # print(f"{label_frq[i]/total_labels},   {np.log(label_frq[i])/np.log(total_labels)}")
  # print()
  # w = 1 - total_labels/label_count[i]
  # w =  np.sqrt(total_labels/label_count[i])
  # w = np.sqrt(1/label_count[i])
  b = 0.9
  w = 1/((1- b**np.log(label_count[i]))/(1-b))
  weights.append(w)
  # print(w)

weights = np.array(weights)
weights = weights / np.sum(weights)
print(weights)



from torch.optim import Adam
# from sklearn.metrics import f1_score   

from datasets import load_metric
metric = load_metric("seqeval")

optimizer = Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    ave_f1 = 0
    cnt =0
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        x = data['x']
        x = x.permute(0, 2, 1)
        y = data['y']

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        loss = loss_function(outputs.view(-1, 5), y.view(-1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            new_y = y.detach().numpy()
            new_outputs = outputs.detach().numpy()
            new_outputs = np.argmax(new_outputs,axis=2)
            results = metric.compute(predictions=new_outputs.reshape(-1).tolist(), references=new_y.reshape(-1).tolist())
            # f1 = f1_score(y_true=new_y.reshape(-1), y_pred=new_outputs.reshape(-1), average='macro') 
            f1 = results['overall_f1']
            ave_f1 += f1
            # f1 = 0
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss} f1-score: {f1}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            cnt += 1

    return last_loss, ave_f1/cnt




from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_f1 = train_one_epoch(epoch_number, writer)
    print(f"Average F1: {avg_f1}")

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    f1 = 0
    for i, vdata in enumerate(test_loader):
        vinputs = vdata['x']
        vinputs = vinputs.permute(0, 2, 1)
        vlabels = vdata['y']
        voutputs = model(vinputs)
        vloss = loss_function(voutputs.view(-1, 5), vlabels.view(-1))
        running_vloss += vloss

        new_vlabels = vlabels.detach().numpy()
        new_voutputs = voutputs.detach().numpy()
        new_voutputs = np.argmax(new_voutputs,axis=2)
        print(new_voutputs.reshape(-1).tolist())
        print(new_vlabels.reshape(-1).tolist())
        results = metric.compute(predictions=new_voutputs.reshape(-1).tolist(), references=new_vlabels.reshape(-1).tolist())
        f1 = results['overall_f1']
        # f1 = f1_score(y_true=new_vlabels.reshape(-1), y_pred=new_voutputs.reshape(-1), average='macro') 

    avg_vloss = running_vloss / (i + 1)
    f1 /=(i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss} F1 valid {f1}')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1




def get_punc(text, splitted=False):
  if not splitted:
    text = text.split()
  x_prepared = get_embedding(text)

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


