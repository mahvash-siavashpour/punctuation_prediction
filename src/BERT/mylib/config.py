from torch import nn



# raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
# train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
# test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'


raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
train_file_name = 'Preprocessed/taaghche/'+'train_taaghche.csv'
test_file_name = 'Preprocessed/taaghche/'+'test_taaghche.csv'



unique_tags = set({'_qMark', '_exMark', 'O', '_dot', '_comma'})
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}




bert_model_name = 'HooshvareLab/distilbert-fa-zwnj-base'
chunksize = 250
train_data_size = 100000000
test_data_size = 20000000



EPOCHS_classifier = 5
LEARNING_RATE = 3.5e-06

EPOCHS_finetune=15

