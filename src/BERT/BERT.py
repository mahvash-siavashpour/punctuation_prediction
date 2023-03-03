# -*- coding: utf-8 -*-
"""9.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1ptCZnCZzaBDVG_DExfnXL3b-FZAedWSa
"""

root = "../../Data/"

import pandas as pd
import torch
import csv
import numpy as np
from sklearn.model_selection import train_test_split

punc = ['.', '،', '؟', '!']
# raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
# train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
# test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'


raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
train_file_name = 'Preprocessed/taaghche/'+'train_taaghche.csv'
test_file_name = 'Preprocessed/taaghche/'+'test_taaghche.csv'


"""# Tokenization

## Costume dataset
"""

unique_tags = set({'_qMark', '_exMark', 'O', '_dot', '_comma'})
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

print(id2tag)

def encode_tags_first(tags, encoding, label_all_tokens = False):
    
    # create an empty array of -100 of length max_length
    encoded_labels = np.ones(len(encoding["attention_mask"]), dtype=int) * -100
    labels = [tag2id[t] for t in tags]
    # print(tag_labels)


    # set only labels whose first offset position is 0 and the second is not 0
    i = 0
    for idx, mapping in enumerate(encoding["offset_mapping"]):
      if mapping[0] == 0 and mapping[1] != 0:
        # overwrite label
        encoded_labels[idx] = labels[i]
        i += 1

    return encoded_labels

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,text, tags, tokenizer, max_len=512):
      
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = text
        self.tags = tags

    def __len__(self):
        return len(self.text)


    def __getitem__(self, index):
        # Generates one sample of data
        # 
        text = self.text[index]
        tags = self.tags[index]
        # print(len(text))
        # print(len(tags))
        # print("------")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(text,
                             is_split_into_words=True,
                             return_offsets_mapping=True, 
                             truncation=True,
                             padding='max_length', 
                             max_length=self.max_len
                             )
        # step3: create encoded labels
        encoded_labels = encode_tags_first(tags, encoding)

        encoding.pop("offset_mapping") # we don't want to pass this to the model

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

"""## Feed tokenizer"""
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from datasets import load_metric
metric = load_metric("seqeval")

bert_model_name = 'HooshvareLab/distilbert-fa-zwnj-base'
chunksize = 250

# tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)

def read_data(file_name, nrows):
  df = pd.read_csv(root+file_name,sep=',', nrows=nrows)
  text = df.iloc[:, 0].values
  tags = df.iloc[:, 1].values
  i = 0 
  input_tokens = []
  input_labels = []

  tmp_tokens = []
  tmp_tags = []

  for word, label in zip(text, tags):
    if i % 250 == 0 and i !=0:
      input_tokens.append(tmp_tokens)
      input_labels.append(tmp_tags)
      tmp_tokens = []
      tmp_tags = []

    tmp_tokens.append(word)
    tmp_tags.append(label)
    i += 1

  return input_tokens, input_labels

train_tokens, train_tags = read_data(train_file_name, 10000000)
test_tokens, test_tags = read_data(test_file_name, 2000000)

# Commented out IPython magic to ensure Python compatibility.
training_set = MyDataset(text=train_tokens, tags=train_tags, tokenizer=tokenizer)
testing_set = MyDataset(text=test_tokens, tags=test_tags, tokenizer=tokenizer)
# %time

print(len(training_set[0]['labels']))

# Parameters

train_params = {'batch_size': 8,
                'shuffle': False,
                'num_workers': 2
                }

test_params = {'batch_size': 8,
                'shuffle': False,
                'num_workers': 2
                }

training_loader = torch.utils.data.DataLoader(training_set, **train_params)
testing_loader = torch.utils.data.DataLoader(testing_set, **test_params)

"""## Loss Scheme"""

label_count = {}
total_labels = 0
for u in id2tag.keys():
  label_count[str(u)] = 0

for idx, batch in enumerate(training_loader):
  labels = batch['labels']
 
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

print(id2tag)

from torch import nn

class_weights = torch.from_numpy(weights).float().to(device)

loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

"""## Defining Costume Model"""


from torch.autograd import Variable 
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DistilBertModel

class CustomModel(nn.Module):
    def __init__(self, num_classes, checkpoint, hidden_dim=768, mlp_dim=100, dropout=0.1, loss_fct=nn.CrossEntropyLoss()):
        super(CustomModel,self).__init__() 

        self.num_labels = num_classes
        self.loss_fct = loss_fct

        #Load Model with given checkpoint and extract its body
        # self.bert = transformers.AutoModel.from_pretrained(checkpoint,config=transformers.AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.bert = DistilBertModel.from_pretrained(bert_model_name, num_labels=len(unique_tags))
        self.dropout = nn.Dropout(dropout) 

        # self.classifier = nn.Linear(768,num_classes)


        input_dim = 768
        hidden_dim = 100
        n_layers = 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.classifier = nn.Linear(mlp_dim, num_classes)
        


        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, mlp_dim),
        #     nn.ReLU(),            
        #     nn.Linear(mlp_dim, num_classes)
        # )


   
    def forward(self, input_ids=None, attention_mask=None,labels=None):
        # output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.bert(   
            input_ids,
            attention_mask=attention_mask,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )


        #Add custom layers
        sequence_output = self.dropout(output[0]) #outputs[0]=last hidden state

        # logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
        
        #LSTM
        logits, (hidden, cell) = self.lstm(sequence_output)
        logits = self.classifier(logits)        


        # mlp
        # logits = self.mlp(sequence_output)
        # print(logits.shape)


        # classifier
        # logits = self.classifier(sequence_output)

        # print(logits.shape)
        # print(logits.reshape(16*512, self.num_labels).shape)

        loss = None
        if labels is not None:
            loss = loss_fct(logits.reshape(logits.shape[0]*logits.shape[1], self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )



model = CustomModel(num_classes=5, checkpoint= bert_model_name, loss_fct=loss_fct)
model.to(device)
print(model)

"""## Training With HuggingFace"""

EPOCHS = 3
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

"""### Huggingface Trainer"""

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

u_tags =list(unique_tags)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [u_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [u_tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }




from torch import nn
from transformers import Trainer

# class_weights = torch.from_numpy(weights).float().to(device)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss (suppose one has 5 labels with different weights)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = loss_fct(logits.reshape(logits.shape[0]*logits.shape[1], self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000,
    save_steps = 1000
)


trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=training_set,         # training dataset
    eval_dataset=testing_set,             # evaluation dataset
    compute_metrics = compute_metrics
)


trainer.train()



"""### To get the precision/recall/f1 computed for each category now that we have finished training, we can apply the same function as before on the result of the predict method:"""

predictions, labels, _ = trainer.predict(testing_set)
predictions = np.argmax(predictions, axis=2)

all_tags = list(unique_tags)
print(all_tags)

# Remove ignored index (special tokens)
true_predictions = [
    [all_tags[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [all_tags[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
