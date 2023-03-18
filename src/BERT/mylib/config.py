import os



def SetModelConfig(model_name, models):

    model_config = models[model_name]

    data_root = "../../Data/"
    
        
    if not os.path.exists(model_config['save_model_path']):
        os.makedirs(model_config['save_model_path'])
        print("Directory " , model_config['save_model_path'] ,  " Created ")
    else:    
        print("Directory " , model_config['save_model_path'] ,  " already exists")


    if not os.path.exists(model_config['log_file_path']):
        os.makedirs(model_config['log_file_path'])
        print("Directory " , model_config['log_file_path'] ,  " Created ")
    else:    
        print("Directory " , model_config['log_file_path'] ,  " already exists")


    if not os.path.exists(model_config['log_file_path']+"Inference/"):
        os.makedirs(model_config['log_file_path']+"Inference/")
        print("Directory " , model_config['log_file_path']+"Inference/" ,  " Created ")
    else:    
        print("Directory " , model_config['log_file_path']+"Inference/" ,  " already exists")


    model_config['save_model_path'] = model_config['save_model_path'] + model_config['model_name'] + "_"+ model_config["model_architecture"]+"_"+model_config['dataset_name'] 
    model_config['log_file_path'] = model_config['log_file_path'] + model_config['model_name'] + "_"+ model_config["model_architecture"]+"_"+model_config['dataset_name']  +".txt"
    model_config['log_file_path_inference'] = model_config['log_file_path'] + "Inference/" + model_config['model_name'] + "_"+ model_config["model_architecture"]+"_"+model_config['dataset_name']  +".txt"


    if model_config['dataset_name'] == "wiki":
        par = ""
        if model_config["include_paragraph_tag"] == "yes":
            par = "paragraph_"
        model_config["raw_data_file_name"] = data_root+ "03_wiki_normalized_tokenized_word_neighbouring.txt"
        model_config["train_file_name"] = data_root+ 'Preprocessed/wiki/'+ par +'train_wiki.csv'
        model_config["test_file_name"] = data_root+ 'Preprocessed/wiki/'+ par +'test_wiki.csv'
    
    elif model_config['dataset_name'] == "taaghche":
        par = ""
        if model_config["include_paragraph_tag"] == "yes":
            par = "paragraph_"
        model_config["raw_data_file_name"] = data_root+ '07_taaghche_v2_normalized_tokenized_word_neighbouring_head200K.txt'
        model_config["train_file_name"] = data_root+ 'Preprocessed/taaghche/'+ par+'train_taaghche.csv'
        model_config["test_file_name"] = data_root+ 'Preprocessed/taaghche/'+par+'test_taaghche.csv'




    if model_config["include_paragraph_tag"] == "no":
        model_config["tag2id"] = {'O': 0,
          'I-dot': 1,
          'I-comma': 2,
          'I-qMark': 3,
          'I-exMark':4
          }
    
    elif model_config["include_paragraph_tag"] == "yes":
        model_config["tag2id"] = {'O': 0,
          'I-dot': 1,
          'I-comma': 2,
          'I-qMark': 3,
          'I-exMark':4,
          'I-par':5
          }
        
    model_config["id2tag"] = {id: tag for tag, id in model_config["tag2id"].items()}
    model_config["unique_tags"] = set(model_config["tag2id"])
 

    return model_config







# bert_model_name = 'HooshvareLab/distilbert-fa-zwnj-base'
# # bert_model_name = 'HooshvareLab/bert-fa-base-uncased'
# chunksize = 100
# train_data_size = 50000000
# test_data_size = 20000000

# seq_max_len = 200


# EPOCHS_classifier = 5
# LEARNING_RATE = 3.5e-06

# EPOCHS_finetune=5
