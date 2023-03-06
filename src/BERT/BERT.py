# -*- coding: utf-8 -*-
"""9.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1ptCZnCZzaBDVG_DExfnXL3b-FZAedWSa
"""

root = "../../Data/"

import torch
import numpy as np

from mylib import bert_train_func
from mylib import dataload_func
from mylib import config

# load Data


"""# Tokenization

## Costume dataset
"""

unique_tags = config.unique_tags
tag2id = config.tag2id
id2tag = config.id2tag

print(id2tag)


"""## Feed tokenizer"""
import transformers
from transformers import DistilBertTokenizerFast
# from transformers import DistilBertForTokenClassification

from datasets import load_metric
metric = load_metric("seqeval")

bert_model_name = config.bert_model_name
chunksize = config.chunksize

# tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)

train_tokens, train_tags = dataload_func.read_data(root+config.train_file_name, config.train_data_size)
test_tokens, test_tags = dataload_func.read_data(root+config.test_file_name, config.test_data_size)

# Commented out IPython magic to ensure Python compatibility.
training_set = dataload_func.MyDataset(text=train_tokens, tags=train_tags, tokenizer=tokenizer)
testing_set = dataload_func.MyDataset(text=test_tokens, tags=test_tags, tokenizer=tokenizer)
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
label_count = dataload_func.label_counts(id2tag, training_loader)
print(label_count)
weights = dataload_func.loss_weights(label_count)
print(weights)


# class_weights = torch.from_numpy(weights).float().to(device)

loss_fct = bert_train_func.loss_fct(weights=None)

"""## Defining Costume Model"""

model = bert_train_func.CustomModel(num_classes=5, checkpoint= bert_model_name, loss_fct=loss_fct, bert_model_name=bert_model_name)
# model.to(device)
print(model)

"""## Training With HuggingFace"""


# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


from transformers import AdamW


for param in model.bert.parameters():
    param.requires_grad = False

pretrained = model.bert.parameters()
# Get names of pretrained parameters (including `bert.` prefix)
pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]

new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]

optimizer = AdamW(
    [{'params': pretrained}, {'params': new_params, 'lr': config.LEARNING_RATE * 10}],
    lr=config.LEARNING_RATE,
)

"""### Huggingface Trainer"""

from transformers import TrainingArguments


# class_weights = torch.from_numpy(weights).float().to(device)

import math
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=config.EPOCHS_classifier,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000,
    save_steps = 1000
)


trainer = bert_train_func.CustomTrainer(
    loss_fct=loss_fct
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=training_set,         # training dataset
    eval_dataset=testing_set,             # evaluation dataset
    compute_metrics = bert_train_func.compute_metrics,
    optimizers = (optimizer, transformers.get_scheduler('linear', optimizer, num_training_steps=math.ceil(len(training_set)/16)* config.EPOCHS_classifier, num_warmup_steps=300))
)

trainer.train()



# fine tuning
for param in model.bert.parameters():
    param.requires_grad = True



import math
training_args2 = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=config.EPOCHS_finetune,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000,
    save_steps = 1000
)

trainer2 = bert_train_func.CustomTrainer(
    loss_fct=loss_fct
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args2,                  # training arguments, defined above
    train_dataset=training_set,         # training dataset
    eval_dataset=testing_set,             # evaluation dataset
    compute_metrics = bert_train_func.compute_metrics,
    optimizers = (optimizer, transformers.get_scheduler('linear', optimizer, num_training_steps=math.ceil(len(training_set)/16)* config.EPOCHS_finetune, num_warmup_steps=300))
)

trainer2.train()


trainer.save_model("../../saved_models/awsome_pp")

trainer.save_model("../../saved_models/awsome_pp1")
torch.save(model.state_dict(), "../../saved_models/awsome_pp2")

# from transformers import DistilBertConfig, DistilBertModel
# path = 'path_to_my_model'
# model = DistilBertModel.from_pretrained(path)

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


# # Inference
# from transformers import pipeline

# text = "ایران سرزمین زیبایی است من در ایران زندگی میکنم آیا ایران هوای خوبی دارد"
# classifier = pipeline("ner", model="../../saved_models/awsome_pp")

# classifier(text)

