from transformers import Trainer
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DistilBertModel
from mylib import config
import numpy as np

from datasets import load_metric
metric = load_metric("seqeval")

u_tags =list(config.unique_tags)


def loss_fct(weights):
    if weights != None:
        loss_fct = nn.CrossEntropyLoss(weights=weights)
        return loss_fct
    else:
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct


class CustomModel(nn.Module):
    def __init__(self, num_classes, checkpoint, bert_model_name, hidden_dim=100, mlp_dim=100, dropout=0.1, loss_fct=nn.CrossEntropyLoss(), model_type='mlp'):
        super(CustomModel,self).__init__() 

        self.num_labels = num_classes
        self.loss_fct = loss_fct
        self.model_type = model_type

        #Load Model with given checkpoint and extract its body
        # self.bert = transformers.AutoModel.from_pretrained(checkpoint,config=transformers.AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.bert = DistilBertModel.from_pretrained(bert_model_name, num_labels=len(config.unique_tags))
        self.dropout = nn.Dropout(dropout) 

        if self.model_type == 'simple_classifier':
            self.classifier = nn.Linear(768,num_classes)

        elif self.model_type == 'lstm':
            input_dim = 768
            n_layers = 1
            self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
            self.classifier = nn.Linear(mlp_dim, num_classes)
                                        

        elif self.model_type == 'mlp':
            input_dim = 768
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),            
                nn.Linear(mlp_dim, num_classes)
            )


   
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

        # classifier
        sequence_output = self.dropout(output[0]) #outputs[0]=last hidden state

        if self.model_type == 'simple_classifier':
            logits = self.classifier(sequence_output) # calculate losses
        
        #LSTM
        elif self.model_type == 'lstm':
            
            logits, (hidden, cell) = self.lstm(sequence_output)
            logits = self.classifier(logits)        


        # mlp
        elif self.model_type == 'mlp':
            logits = self.mlp(sequence_output)


        

        # print(logits.shape)
        # print(logits.reshape(16*512, self.num_labels).shape)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.reshape(logits.shape[0]*logits.shape[1], self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


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


class CustomTrainer(Trainer):
    def __init__(self,loss_fct, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.loss_fct = loss_fct

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        # compute custom loss (suppose one has 5 labels with different weights)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = self.loss_fct(logits.reshape(logits.shape[0]*logits.shape[1], self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
