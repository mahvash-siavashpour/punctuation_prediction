import torch
import argparse
import json
import sys
import numpy as np
# import language_lists


from mylib import bert_train_func
from mylib import config

parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')
args = parser.parse_args()


### read models configuration json file
with open("bert_models.json") as f:
    models = json.load(f)
    models_name = list(models.keys())


configurations = config.SetModelConfig(args.model_name, models)

sys.stdout = open(configurations["log_file_path_inference"], 'w', encoding="utf-8")


unique_tags = configurations["unique_tags"]
tag2id = configurations["tag2id"]
id2tag = configurations["id2tag"]


bert_model_name = configurations["bert_model_name"]
chunksize = configurations["chunksize"]
loss_fct = bert_train_func.loss_fct(weights=None)


model = bert_train_func.CustomModel(num_classes=5, checkpoint= bert_model_name, loss_fct=loss_fct, bert_model_name=bert_model_name)


model.load_state_dict(torch.load(configurations["save_model_path"]))
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
        if tag == 'I-comma':
            result.append("،")
        elif tag == 'I-dot':
            result.append(".")
        elif tag == 'I-qMark':
            result.append("؟")
        elif tag == 'I-exMark':
            result.append("!")
        elif tag == "I-par":
           result.append(".\n")
    return result



def bert_get_punc(text, tokenizer, id2tag, is_splitted=False, max_length=configurations["bert_seq_max_len"]):
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
  new_output_text = " ".join(output_text)
                   
  return final_results, new_output_text




from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)




text1 = "بعضی از ما یادمان می‌آید که در دوران نوجوانی خود در انتخاب هر چیزی مشکل داشتیم گاه حتی نمی‌دانستیم چه چیزی را دوست داریم و چه چیزی را دوست نداریم این مسئله اکنون در کودکان‌مان قابل‌مشاهده است کودکان ما نیز در شرایط مشابه یعنی انتخاب رنگ لباس نوع غذا و غیره همان مشکلاتی را دارند که ما زمانی داشتیم"
out, output_text = bert_get_punc(text=text1, tokenizer=tokenizer, id2tag=id2tag)

print(out)

print("--------------------------------------------------")

text2 = "ایران سرزمین زیبایی است من در ایران زندگی میکنم آیا ایران هوای خوبی دارد"

out, output_text = bert_get_punc(text=text2, tokenizer=tokenizer, id2tag=id2tag)

print(out)
print(output_text)



