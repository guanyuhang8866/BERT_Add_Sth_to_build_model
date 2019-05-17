#!/usr/bin/python3

#@File: main.py

#-*-coding:utf-8-*-

#@Author:Guan_pc

#@Time: 2019年05月15日16时

#说明: 运行此文件得到feed给模型的训练和测试数据

import codecs
import pickle

from keras_bert import Tokenizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

train1 = [i.strip() for i in open('CuttedWord_title.txt', 'r', encoding='utf8')][:-1]
label = [i.strip() for i in open('label.txt', 'r', encoding='utf8')][:-1]

dict_path = "../chinese_L-12_H-768_A-12/vocab.txt"

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
x1, x2 = [], []
for text in train1:
    text_ = text.replace(" ","")[:28]
    indices, segments = tokenizer.encode(first=text_, max_len=512)
    x1.append(indices)
    x2.append(segments)
lb = LabelBinarizer()
y = lb.fit_transform(label)
joblib.dump(lb,'lb.model')
x1, validate_x1, x2, validate_x2, y, validate_y = train_test_split(x1, x2, y, test_size=0.01, random_state=255)
with open('bert_title.pkl','wb') as f:
    pickle.dump(x1,f)
    pickle.dump(validate_x1, f)
    pickle.dump(x2, f)
    pickle.dump(validate_x2, f)
    pickle.dump(y, f)
    pickle.dump(validate_y, f)
