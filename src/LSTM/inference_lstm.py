import torch
import numpy as np
from mylib import lstm_train_func
from mylib import dataload_func
from mylib import config
import argparse
import json
import sys


import torch
import numpy as np
import json
# import fasttext
from gensim.models import fasttext

from mylib import config, lstm_train_func, dataload_func


parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')

args = parser.parse_args()


def lstm_get_punc(text, model_name, splitted=False):
    with open("lstm_modelsjson") as f:
        models = json.load(f)

    configurations = config.SetModelConfig(model_name, models)

    unique_tags = configurations["unique_tags"]
    tag2id = configurations["tag2id"]
    id2tag = configurations["id2tag"]

    model = lstm_train_func.LSTM_Model(input_size=configurations['input_size'],
                                       hidden_size=configurations['lstm_hidden_size'],
                                       num_layers=1, num_classes=len(list(unique_tags)),
                                       use_cnn=configurations['use_cnn'])

    model.load_state_dict(torch.load("models/LSTM/"+configurations["save_model_name"], map_location=torch.device('cpu')))
    model.eval()

    if not splitted:
        text = text.split()

    # fasttext_model = fasttext.load_model('ml_scripts/LSTM/word-embeddings/cc.fa.300.bin')
    fasttext_model = fasttext.load_facebook_model('word-embeddings/cc.fa.300.bin', encoding='utf-8')
    x_prepared = dataload_func.get_embedding(text, fasttext_model)

    X = torch.from_numpy(x_prepared).float()
    print(X.shape)
    # X = X.reshape(1, 300, 10)
    # out = model(X)
    # out = out.detach().numpy()
    # new_outputs = np.argmax(out, axis=2)

    # result = []
    # for o, t in zip(new_outputs[0], text):
    #     result.appen  d((t, id2tag[o]))

    # return result


text = "من در ایران زندگی میکنم ولی شما چطور زندگی میکنید"
print(lstm_get_punc(text, args.model_name))

text2 = "جشن باستانی نوروز ایرانی پیشینه جالب و خواندنی‌ای دارد با کمک روایت‌های فردوسی حکیم در شاهنامه می‌توانیم داستان نوروز را بفهمیم ماجرا از این قرار است که در تمام کتاب‌های تاریخی از جمله شاهنامه فردوسی در کنار نوروز نام جمشید آمده است همه این روز را به زمان پادشاهی او نسبت داده‌اند و او را پایه‌گذار جشن نوروز می‌دانند اعتقاد بر این است که نوروز چون همزمان با آمدن فصل بهار و تولد دوباره طبیعت است انسان هم در این روز تولدی دوباره می‌یابد و به‌عنوان جزیی از هستی و عالم آفرینش مانند وجودی تازه متولد شده بی‌گناه و پاکیزه است تاریخچه و آداب و رسوم عید نوروز این روز در ایران و افغانستان نوید‌ دهنده سال جدید است "
print(lstm_get_punc(text2, args.model_name))
