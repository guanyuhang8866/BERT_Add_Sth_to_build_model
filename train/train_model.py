#!/usr/bin/python3

#@File: train_model.py

#-*-coding:utf-8-*-

#@Author:Guan_pc

#@Time: 2019年05月16日09时

# 说明:用官方的bert模型和各种特征提取器结合成新模型并训练特征提取器的参数
import pickle

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint

from train.models import *

with open("../data/bert_title.pkl","rb") as f:
    x1 = pickle.load(f)
    validate_x1 = pickle.load(f)
    x2 = pickle.load(f)
    validate_x2 = pickle.load(f)
    y = pickle.load(f)
    validate_y = pickle.load(f)

def slice(x, index):
    return x[:, :index, :]

config_path = "../chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "../chinese_L-12_H-768_A-12/bert_model.ckpt"


bert_embeding = load_trained_model_from_checkpoint(config_path, checkpoint_path)
for layer in bert_embeding.layers:
    layer.trainable = False
lamb = Lambda(slice,arguments={'index':30})(bert_embeding.output)
out1 = textCNNfeature(lamb,"out_cont")
output = Dense(5, activation='softmax')(out1)
model = Model(inputs=bert_embeding.input, outputs=output)
model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['acc'])
model.summary()
plot_model(model)


save_best = ModelCheckpoint('model_V3_cnn.h5', verbose=1, save_best_only=True, save_weights_only=True)
model.fit(x={'Input-Token': np.array(x1), 'Input-Segment': np.array(x2)}, y= y,
          batch_size=64, epochs=10,verbose=1,
          validation_data=({'Input-Token': np.array(validate_x1), 'Input-Segment': np.array(validate_x2)}, validate_y ),
          callbacks = [save_best]
          )