import pandas as pd
import torch
import numpy as np



def encode_tags_first(tags, encoding, tag2id, label_all_tokens = False):
    
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
    def __init__(self,text, tags, tokenizer, tag2id, max_len=512):
      
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = text
        self.tags = tags
        self.tag2id = tag2id

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
        encoded_labels = encode_tags_first(tags, encoding, self.tag2id)

        encoding.pop("offset_mapping") # we don't want to pass this to the model

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item
    


def read_data(file_name, nrows, chunksize):
  df = pd.read_csv(file_name,sep=',', nrows=nrows)
  text = df.iloc[:, 0].values
  tags = df.iloc[:, 1].values
  i = 0 
  input_tokens = []
  input_labels = []

  tmp_tokens = []
  tmp_tags = []

  for word, label in zip(text, tags):
    if i % chunksize == 0 and i !=0:
      input_tokens.append(tmp_tokens)
      input_labels.append(tmp_tags)
      tmp_tokens = []
      tmp_tags = []

    tmp_tokens.append(word)
    tmp_tags.append(label)
    i += 1

  return input_tokens, input_labels




def label_counts(id2tag, training_loader):
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

    return label_count


def loss_weights(label_count):
    weights = []
    for i in label_count.keys():
        # print(f"{label_frq[i]/total_labels},   {np.log(label_frq[i])/np.log(total_labels)}")
        # print()
        # w = 1 - total_labels/label_count[i]
        # w =  np.sqrt(total_labels/label_count[i])
        # w = np.sqrt(1/label_count[i])
        b = 0.9
        w = 1/float(((1- b**np.log(label_count[i]))/(1-b)))
        weights.append(w)
        # print(w)

    weights = np.array(weights)
    weights = weights / np.sum(weights)
        
    return weights