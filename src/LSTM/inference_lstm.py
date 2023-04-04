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
import fasttext
# from gensim.models import fasttext

from mylib import config, lstm_train_func, dataload_func


parser = argparse.ArgumentParser(description='Makes Predition for Punctuation Marks')
parser.add_argument('model_name',
                    help='Model Name')

args = parser.parse_args()

with open("lstm_models.json") as f:
    models = json.load(f)

configurations = config.SetModelConfig(args.model_name, models)


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


def lstm_get_punc(text, splitted=False):


    

    unique_tags = configurations["unique_tags"]
    tag2id = configurations["tag2id"]
    id2tag = configurations["id2tag"]

    model = lstm_train_func.LSTM_Model(input_size=configurations['input_size'],
                                       hidden_size=configurations['lstm_hidden_size'],
                                       num_layers=1, num_classes=len(list(unique_tags)),
                                       use_cnn=configurations['use_cnn'])

    model.load_state_dict(torch.load(configurations["save_model_path"]))
    model.eval()

    if not splitted:
        text = text.split()

    fasttext_model = fasttext.load_model('word-embeddings/cc.fa.300.bin')
    # fasttext_model = fasttext.load_facebook_model('word-embeddings/cc.fa.300.bin', encoding='utf-8')
    x_prepared = dataload_func.get_embedding(text, fasttext_model)

    X = torch.from_numpy(x_prepared).float()
    # print(X.shape)
    X = X.reshape(1, 300, X.shape[0])
    out = model(X)
    out = out.detach().numpy()
    new_outputs = np.argmax(out, axis=2)
    print(new_outputs.shape)

    result = []
    for o, t in zip(new_outputs[0], text):
        result.append((t, id2tag[o]))

    output_text = insert_punc(result)
    new_output_text = " ".join(output_text)

    return result, new_output_text


sys.stdout = open(configurations["log_file_path_inference"], 'w', encoding="utf-8")

text1 = "نوروز نخستین روز سال خورشیدی ایرانی برابر با یکم فروردین ماه جشن آغاز سال نوی ایرانی و یکی از کهن‌ترین جشن‌های به جا مانده از دوران ایران باستان است خاستگاه نوروز در ایران باستان است و در مناطق وسیعی در آسیا و دیگر نقاط جهان جشن گرفته می‌شود زمان برگزاری نوروز اعتدال بهاری و در آغاز فصل بهار است نوروز در ایران و افغانستان آغاز سال نو محسوب می‌شود و در برخی دیگر از کشورها یعنی تاجیکستان روسیه قرقیزستان قزاقستان سوریه عراق گرجستان جمهوری آذربایجان آلبانی چین ترکمنستان هند پاکستان و ازبکستان تعطیل رسمی است و مردمان آن جشن را برپا می‌کنند"
print(lstm_get_punc(text1))

text2 = "جشن باستانی نوروز ایرانی پیشینه جالب و خواندنی‌ای دارد با کمک روایت‌های فردوسی حکیم در شاهنامه می‌توانیم داستان نوروز را بفهمیم ماجرا از این قرار است که در تمام کتاب‌های تاریخی از جمله شاهنامه فردوسی در کنار نوروز نام جمشید آمده است همه این روز را به زمان پادشاهی او نسبت داده‌اند و او را پایه‌گذار جشن نوروز می‌دانند اعتقاد بر این است که نوروز چون همزمان با آمدن فصل بهار و تولد دوباره طبیعت است انسان هم در این روز تولدی دوباره می‌یابد و به‌عنوان جزیی از هستی و عالم آفرینش مانند وجودی تازه متولد شده بی‌گناه و پاکیزه است تاریخچه و آداب و رسوم عید نوروز این روز در ایران و افغانستان نوید‌ دهنده سال جدید است "
print(lstm_get_punc(text2))

text3="نوروز نخستین روز سال خورشیدی ایرانی برابر با یکم فروردین ماه جشن آغاز سال نوی ایرانی و یکی از کهن‌ترین جشن‌های به جا مانده از دوران ایران باستان است خاستگاه نوروز در ایران باستان است و در مناطق وسیعی در آسیا و دیگر نقاط جهان جشن گرفته می‌شود فعالیت صنعت خودروسازی ایران با مونتاژ خودروهایی همچون خودروهای نظامی و شهری جیپ کلید خورد اما طولی نکشید که خودروسازان ایرانی از چرخه مونتاژ خارج و به تولیدکننده خودرو تبدیل شدند دومین کارخانه‌ خودروسازی ایران سایپا بود که در سال ۱۳۴۴ تاسیس شد اولین خودرویی که در سایپا تولید شد ژیان بود که با مشارکت سیتروئن فرانسه تولید می‌شد ژیان مهاری ژیان پیکاپ رنو ۵ و رنو ۲ از خودروهایی بودند که توسط سایپا تا سال ۵۶ تولید شدند"
print(lstm_get_punc(text3))