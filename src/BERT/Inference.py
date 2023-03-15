import pandas as pd
import copy
from tqdm import tqdm
import torch
import os
import re
import csv
import numpy as np
# import language_lists


from mylib import bert_train_func
from mylib import config


bert_model_name = config.bert_model_name
chunksize = config.chunksize
loss_fct = bert_train_func.loss_fct(weights=None)


model = bert_train_func.CustomModel(num_classes=5, checkpoint= bert_model_name, loss_fct=loss_fct, bert_model_name=bert_model_name)


model.load_state_dict(torch.load('../../saved_models/BERT/pp_bert'), strict=False)
model.eval()

print(model)


def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
 

def insert_punc(output):
    result = []

    for (token, tag) in output[0]:
        result.append(token)
        if tag == '_comma':
            result.append("،")
        elif tag == '_dot':
            result.append(".")
        elif tag == '_qMark':
            result.append("؟")
        elif tag == '_exMark':
            result.append("!")
    return result



def bert_get_punc(text, tokenizer, id2tag, is_splitted=False, max_length=512):
  if not is_splitted:
    text = text.split()


  chunk_text = list(divide_chunks(text, int(max_length/2)))
  final_results = []
  for ct in chunk_text:
    encoding = tokenizer(ct,
                        is_split_into_words=True,
                        return_offsets_mapping=True, 
                        truncation=True,
                        padding='max_length', 
                        max_length=max_length,
                        return_tensors="pt")
    
    input_encoding = encoding["input_ids"]

    out = model(input_encoding)
    out=out['logits']

    outputs = out.detach().cpu().numpy()
    new_outputs = np.argmax(outputs,axis=2)


    encoded_labels = np.ones(len(encoding["attention_mask"][0]), dtype=int) * -100

    # set only labels whose first offset position is 0 and the second is not 0
    i = 0
    for idx, mapping in enumerate(encoding["offset_mapping"][0]):
      if mapping[0] == 0 and mapping[1] != 0:
        # overwrite label
        encoded_labels[idx] = 0
        i += 1

    true_predictions1 = []
    for (p, l) in zip(new_outputs[0], encoded_labels):
      if l != -100:
        true_predictions1.append(id2tag[p])
    
    result = [] 
    for o, t in zip(true_predictions1, ct):
      result.append((t, o))

    final_results.append(result)

  output_text = insert_punc(final_results)
                         
  return final_results, output_text




from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_name)


id2tag = config.id2tag


text1 = "بعضی از ما یادمان می‌آید که در دوران نوجوانی خود در انتخاب هر چیزی مشکل داشتیم گاه حتی نمی‌دانستیم چه چیزی را دوست داریم و چه چیزی را دوست نداریم این مسئله اکنون در کودکان‌مان قابل‌مشاهده است کودکان ما نیز در شرایط مشابه یعنی انتخاب رنگ لباس نوع غذا و غیره همان مشکلاتی را دارند که ما زمانی داشتیم"
out, output_text = bert_get_punc(text=text1, tokenizer=tokenizer, id2tag=id2tag)

print(out)
new_output_text = " ".join(output_text)
print(" ".join(new_output_text))




