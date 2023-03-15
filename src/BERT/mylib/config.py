from torch import nn



# raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
# train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
# test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'


raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
train_file_name = 'Preprocessed/taaghche/'+'train_taaghche.csv'
test_file_name = 'Preprocessed/taaghche/'+'test_taaghche.csv'



unique_tags = set({'I-qMark', 'I-exMark', 'O', 'I-dot', 'I-comma'})
tag2id = {'O': 0,
          'I-dot': 1,
          'I-comma': 2,
          'I-qMark': 3,
          'I-exMark':4
          }
id2tag = {id: tag for tag, id in tag2id.items()}



bert_model_name = 'HooshvareLab/distilbert-fa-zwnj-base'
# bert_model_name = 'HooshvareLab/bert-fa-base-uncased'
chunksize = 100
train_data_size = 1000000000
test_data_size = 200000000

max_len = 200


EPOCHS_classifier = 5
LEARNING_RATE = 3.5e-06

EPOCHS_finetune=5

