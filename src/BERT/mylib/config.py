from torch import nn



raw_data_file_name = "03_wiki_normalized_tokenized_word_neighbouring.txt"
train_file_name = 'Preprocessed/wiki/'+'train_wiki.csv'
test_file_name = 'Preprocessed/wiki/'+'test_wiki.csv'


# raw_data_file_name = '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
# train_file_name = 'Preprocessed/taaghche/'+'train_taaghche.csv'
# test_file_name = 'Preprocessed/taaghche/'+'test_taaghche.csv'



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
train_data_size = 50000000
test_data_size = 20000000

max_len = 200


EPOCHS_classifier = 5
LEARNING_RATE = 3.5e-06

EPOCHS_finetune=5




# def SetModelAndPaths(model_name, models):
#     global model_config
#     global plots_path
#     global dataset_path
#     global stan_file_path
#     global stan_output_dir
#     model_config = models[model_name]
#     plots_path = plots_root + model_config['plots_folder_name'] + '/'
#     dataset_path = datasets_root + model_config['dataset_name']
#     stan_file_path = stan_files_root + model_config['stan_file']
#     stan_output_dir = saved_models_root + model_config['model_name'] + '/'
#     os.path
    
#     if not os.path.exists(plots_path):
#         os.makedirs(plots_path)
#         print("Directory " , plots_path ,  " Created ")
#     else:    
#         print("Directory " , plots_path ,  " already exists")
        
#     if not os.path.exists(stan_output_dir):
#         os.makedirs(stan_output_dir)
#         print("Directory " , stan_output_dir ,  " Created ")
#     else:    
#         print("Directory " , stan_output_dir ,  " already exists")


#     return 
