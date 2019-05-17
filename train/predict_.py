#!/usr/bin/python3

#@File: predict_.py

#-*-coding:utf-8-*-

#@Author:Guan_pc

#@Time: 2019年05月16日11时

#说明:测试模型在测试集的正确率
import codecs
import os
import re
import numpy as np
import jieba
import pandas as pd
import xlrd
from keras import Model
from keras.layers import concatenate, Input, Dense, Lambda
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, f1_score

from train.models import *

def slice(x, index):
    return x[:, :index, :]


class Model_V3(object):

    def __init__(self,Sth):
        self.model = self.loadmodel(Sth)
        self.lb = joblib.load("../data/lb.model")
        self.restr = '[\s+\/,%^*\-+]+|[+——~@#%……&*]+'
    def loadmodel(self,Sth):


        config_path = "../chinese_L-12_H-768_A-12/bert_config.json"
        checkpoint_path = "../chinese_L-12_H-768_A-12/bert_model.ckpt"

        bert_embeding = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        lamb = Lambda(slice, arguments={'index': 30})(bert_embeding.output)
        out1 = Sth(lamb, "out_cont")
        output = Dense(5, activation='softmax')(out1)
        model = Model(inputs=bert_embeding.input, outputs=output)
        model.load_weights("model_V3_cnn.h5")
        return model

    def work1(self, text1):
        out = []
        if type(text1) == str:
            text1 = [text1]
        for i in text1:
            resu = i.replace('|', '').replace('&nbsp;', '').replace('&ldquo;', '“').replace('&rdquo;', '”') \
                .replace('&lsquo;', '‘').replace('&rsquo;', '’').replace('〔', '（').replace('〕', '）').replace('/', '') \
                .replace('&middot;', '·').replace('•', '·').replace("\\n", "\n").replace("\\r", "\r").replace("\\t",
                                                                                                              "\t")
            resu = re.split(r'\s+', resu)
            dr = re.compile(r'<[^>]+>', re.S)
            dd = dr.sub('', '。'.join(resu))
            line = re.sub(self.restr, '', dd)
            eng = [",", "!", "?", ":", ";", "(", ")", "[", "]", "$", "。。"]
            chi = ["，", "！", "？", "：", "；", "（", "）", "【", "】", "￥", '。']
            for i, j in zip(eng, chi):
                line = line.replace(i, j)
            out.append(line[:28])
        token_dict = {}
        dict_path = "../chinese_L-12_H-768_A-12/vocab.txt"
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        tokenizer = Tokenizer(token_dict)
        x1, x2 = [], []
        for text in out:
            indices, segments = tokenizer.encode(first=text, max_len=512)
            x1.append(indices)
            x2.append(segments)
        return x1, x2


    def predict(self, title):
        x1, x2 = self.work1(title)
        pre = self.model.predict([np.array(x1),np.array(x2)], batch_size=64)
        P = [max(i) for i in pre.tolist()]
        pre_y = [i.index(max(i)) for i in pre.tolist()]
        pre_y = [self.lb.classes_[i]for i in pre_y]
        return pre_y,P

if __name__ == '__main__':

    model = Model_V3(Sth = textCNNfeature)
    title = "梦湖孔雀城丨离繁华不远，离自然很近"
    pre, P = model.predict(title)
    print("预测:",pre,"可信度：",P)