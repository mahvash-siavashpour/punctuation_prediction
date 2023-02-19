# -*- coding: utf-8 -*-
"""9.1_punc_pred


Original file is located at
    https://colab.research.google.com/drive/1ptCZnCZzaBDVG_DExfnXL3b-FZAedWSa
"""

root = "/Data/"

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