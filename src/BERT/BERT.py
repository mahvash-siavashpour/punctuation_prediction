# -*- coding: utf-8 -*-
"""9.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1ptCZnCZzaBDVG_DExfnXL3b-FZAedWSa
"""


import torch
import numpy as np

from mylib import bert_train_func
from mylib import dataload_func
from mylib import config
import argparse
import json
import sys



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


# load Data


"""# Tokenization

## Costume dataset
"""

unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]

print(f"tags used in this session: \n{id2tag}")


"""## Feed tokenizer"""
import transformers
from transformers import DistilBertTokenizerFast
# from transformers import DistilBertForTokenClassification

# from datasets import load_metric
# metric = load_metric("seqeval")

bert_model_name = configurations["bert_model_name"]
chunksize = configurations["chunksize"]

# tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)

train_tokens, train_tags = dataload_func.read_data(configurations["train_file_name"], configurations["train_data_size"])
test_tokens, test_tags = dataload_func.read_data(configurations["test_file_name"], configurations["test_data_size"])

# Commented out IPython magic to ensure Python compatibility.
training_set = dataload_func.MyDataset(text=train_tokens, tags=train_tags, tokenizer=tokenizer, max_len=configurations["bert_seq_max_len"])
testing_set = dataload_func.MyDataset(text=test_tokens, tags=test_tags, tokenizer=tokenizer, max_len=configurations["bert_seq_max_len"])
# %time

print(f"length of the sequence: {len(training_set[0]['labels'])}")

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
print(f"number of each labels: \n {label_count}")
weights = dataload_func.loss_weights(label_count)
print(f"weigh of each label: {weights}")


# class_weights = torch.from_numpy(weights).float().to(device)

loss_fct = bert_train_func.loss_fct(weights=None)

"""## Defining Costume Model"""

model = bert_train_func.CustomModel(num_classes=len(list(unique_tags)), loss_fct=loss_fct, bert_model_name=bert_model_name, model_type=configurations["model_architecture"])
# model.to(device)
print(model)

"""## Training With HuggingFace"""


# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


from transformers import AdamW




pretrained = model.bert.parameters()
# Get names of pretrained parameters (including `bert.` prefix)
pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]

new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]

optimizer = AdamW(
    [{'params': pretrained}, {'params': new_params, 'lr': configurations["LEARNING_RATE"] * 10}],
    lr=configurations["LEARNING_RATE"],
)

"""### Huggingface Trainer"""

from transformers import TrainingArguments


# class_weights = torch.from_numpy(weights).float().to(device)

import math

if configurations["pre_tune"] == "yes":
    for param in model.bert.parameters():
        param.requires_grad = False

    training_args1 = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=configurations["EPOCHS_classifier"],              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=300,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,
        save_steps = 1000
    )


    trainer1 = bert_train_func.CustomTrainer(
        loss_fct=loss_fct,
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args1,                  # training arguments, defined above
        train_dataset=training_set,         # training dataset
        eval_dataset=testing_set,             # evaluation dataset
        compute_metrics = bert_train_func.compute_metrics,
        optimizers = (optimizer, transformers.get_scheduler('linear', optimizer, num_training_steps=math.ceil(len(training_set)/16)* configurations["EPOCHS_classifier"], num_warmup_steps=300))
    )

    trainer1.train()



# fine tuning
if configurations["fine_tune"] == "yes":
    for param in model.bert.parameters():
        param.requires_grad = True


    training_args2 = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=configurations["EPOCHS_finetune"],              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=300,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,
        save_steps = 1000
    )

    trainer2 = bert_train_func.CustomTrainer(
        loss_fct=loss_fct,
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args2,                  # training arguments, defined above
        train_dataset=training_set,         # training dataset
        eval_dataset=testing_set,             # evaluation dataset
        compute_metrics = bert_train_func.compute_metrics,
        optimizers = (optimizer, transformers.get_scheduler('linear', optimizer, num_training_steps=math.ceil(len(training_set)/16)* configurations["EPOCHS_finetune"], num_warmup_steps=300))
    )

    trainer2.train()


# trainer.save_model("../../saved_models/awsome_pp1")

# from transformers import DistilBertConfig, DistilBertModel
# path = 'path_to_my_model'
# model = DistilBertModel.from_pretrained(path)

"""### To get the precision/recall/f1 computed for each category now that we have finished training, we can apply the same function as before on the result of the predict method:"""



if configurations["pre_tune"] == "no":
    trainer = trainer2
elif configurations["fine_tune"] == "no":
    trainer = trainer1


#prediction on test set
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

results = bert_train_func.compute_metrics((true_predictions, true_labels))
print(f"**Testing Set Results** \n {results}")





#prediction on train set
predictions_t, labels_t, _ = trainer.predict(training_set)
predictions_t = np.argmax(predictions_t, axis=2)

all_tags = list(unique_tags)
print(all_tags)

# Remove ignored index (special tokens)
true_predictions_t = [
    [all_tags[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions_t, labels_t)
]
true_labels_t = [
    [all_tags[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions_t, labels_t)
]

results_t = bert_train_func.compute_metrics((true_predictions_t, true_labels_t))

print(f"**Training Set Results** \n {results_t}")






#save the model
torch.save(model.state_dict(), "../../saved_models/BERT/pp_bert")
