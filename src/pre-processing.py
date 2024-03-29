# -*- coding: utf-8 -*-
"""9.1_punc_pred_manual.ipynb


Original file is located at
    https://colab.research.google.com/drive/1ptCZnCZzaBDVG_DExfnXL3b-FZAedWSa
"""

root = "../Data/"

import torch
import csv
import language_lists
import argparse
import os
'''
The code must be run in this format:

python3 pre-processing.py [par/no_par] [wiki/taaghche]
'''


parser = argparse.ArgumentParser(description='Preprocessing')
parser.add_argument('paragraph',
                    help='if you want to consider paragraphs: options are par and no_par')

parser.add_argument('dataset_name',
                    help='name of the dataset')

args = parser.parse_args()

dataset_name = args.dataset_name
has_paragraph = False
par = ""

if args.paragraph == 'par':
  has_paragraph = True
  par = "paragraph_"


punc = ['.', '،', '؟', '!']

if not os.path.exists(root+'Preprocessed/'+dataset_name+'/'):
    os.makedirs(root+'Preprocessed/'+dataset_name+'/')
    print("Directory " , root+'Preprocessed/'+dataset_name+'/' ,  " Created ")
else:    
    print("Directory " , root+'Preprocessed/'+dataset_name+'/' ,  " already exists")




if dataset_name == "wiki":
  raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"

elif dataset_name == "taaghche":
  raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'



train_file_name = 'Preprocessed/'+dataset_name+'/'+par+'train_'+dataset_name+'.csv'
test_file_name = 'Preprocessed/'+dataset_name+'/'+par+'test_'+dataset_name+'.csv'

"""# Pre Processing

## Define Functions
"""



def read_raw_data(file):
  f = open(root+file,'r', encoding="utf-8")
  contents = f.read()
  contents = bytes(contents, 'utf-8').decode('utf-8','ignore')
  tokenized_data = contents.split()
  return tokenized_data

def remove_english_chars(input_data):
  flag =  False
  tokenized_content = []
  for idx, i  in enumerate(input_data):
    # some times the punc word and the word are stuck together so we seperate them
    for x in language_lists.all_punctuations:
      if x in i and x != i:
        i = i.replace(x, f" {x} ")
        i = i.split()
        tokenized_content += i
        flag = True
        break
        
    if not flag:
      tokenized_content.append(i)
    flag = False

  
  clean_data= []
  
  for idx, text in enumerate(tokenized_content):
    flag = False
    if '\u200c' not in text:
      for en in language_lists.englsih_chars:
        if en in text.lower():
          flag = True
          break
    if not flag:
      clean_data.append(text)

  return clean_data

def remove_extra_punc(input_data):

  clean_data= []
  for idx, text in enumerate(input_data):
    flag = False
    for p in language_lists.punctuations:
      if p == text:
        flag = True
        break
      
    if not flag:
      clean_data.append(text)

  return clean_data

def seperate_data_and_labels(input_data):
  tags = []
  index = 0

  splitted_text = []
  splitted_tags = []

  # new_chunk_text = []
  # new_chunk_tags = []

  while index < len(input_data) -1:
    token = input_data[index]

    next_token = input_data[index+1]
    if next_token in punc:
      splitted_text.append(token)
      pun_mark = ""
      if has_paragraph:
        if index == len(input_data) -2:
          pun_mark = "I-par"
          splitted_tags.append(pun_mark)

      if next_token == '؟':
        pun_mark = "I-qMark"
        splitted_tags.append(pun_mark)

      elif next_token == '.':
        pun_mark = "I-dot"
        splitted_tags.append(pun_mark)
    
      elif next_token == '!':
        pun_mark = "I-exMark"
        splitted_tags.append(pun_mark)
      
      elif next_token == '،':
        pun_mark = "I-comma"
        splitted_tags.append(pun_mark)
      index += 1
    else:
      splitted_text.append(token)
      splitted_tags.append('O')
    index += 1


  return splitted_text, splitted_tags

def chunke_data(splitted_text, splitted_tags, chunk_size):
  
  chunked_text = []
  chunked_tags = []

  new_chunk_text = []
  new_chunk_tags = []

  for text, tags in zip(splitted_text, splitted_tags):
    if len(new_chunk_text) + 1  <= chunk_size:
      new_chunk_text.append(text)
      new_chunk_tags.append(tags)
    
    else:
      chunked_text.append(new_chunk_text)
      chunked_tags.append(new_chunk_tags)
      new_chunk_text = []
      new_chunk_tags = []
      new_chunk_text.append(text)
      new_chunk_tags.append(tags)

  return chunked_text, chunked_tags

# def preprocess_data(file):
#   data = read_raw_data(file)
#   data = remove_english_chars(data)
#   data = remove_extra_punc(data)
#   splitted_text, splitted_tags = seperate_data_and_labels(data)
#   # chunked_text, chunked_tags = chunke_data(splitted_text, splitted_tags, 250)

#   return splitted_text, splitted_tags

class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self,file_name, chunksize):
        # self.data_files = os.listdir()
        # self.data_files = self.data_files.sort()
        self.datapath = root+file_name
        self.chunksize = chunksize
        self.reader = open(root+file_name,'r', encoding="utf-8")
        self.file_name = file_name
        self.len = self.claculate_len()
        


    def claculate_len(self):
      rows = 0
      for line in open(root+self.file_name,'r', encoding="utf-8"):
        rows += 1

      return rows


    def __len__(self):  
      return self.len



    def __getitem__(self, index):
        # Generates one sample of data
        
      
        data = self.reader.readline()
        # contents = bytes(data, 'utf-8').decode('utf-8','ignore')
        tokenized_data = data.split()
        tokenized_data = remove_english_chars(tokenized_data)
        tokenized_data = remove_extra_punc(tokenized_data)
        splitted_text, splitted_tags = seperate_data_and_labels(tokenized_data)
        # chunked_text, chunked_tags = chunke_data(splitted_text, splitted_tags, 250)

        return splitted_text, splitted_tags

"""## Run"""

preprocessed_dataset = PreprocessDataset(raw_data_file_name, chunksize=100000)

dataset_len = len(preprocessed_dataset)

dataset_len

train_file_handle =  open(root+train_file_name, 'w', encoding='utf-8')
train_file_writer = csv.writer(train_file_handle)
test_file_handle =  open(root+test_file_name, 'w', encoding='utf-8')
test_file_writer = csv.writer(test_file_handle)

for idx, (splitted_text, splitted_tags) in enumerate(preprocessed_dataset):
  if idx == dataset_len:
    break
  if idx > .8*dataset_len:
  # write a row to the csv file
    for text, label in zip(splitted_text, splitted_tags):
      test_file_writer.writerow((text, label))

  else:
    for text, label in zip(splitted_text, splitted_tags):
      train_file_writer.writerow((text, label))

train_file_handle.close()
test_file_handle.close()

