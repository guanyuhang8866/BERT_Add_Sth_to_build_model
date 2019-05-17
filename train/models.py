from keras.layers import *
from keras.models import Model,Sequential
from keras.utils import plot_model
from keras_pos_embd import TrigPosEmbedding
from keras_transformer import get_encoders
from sklearn.model_selection import KFold

def textCNNfeature(input,name):

    pool_output = []
    kernel_sizes = [2, 3, 4, 5]
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=32, kernel_size=kernel_size, strides=1)(input)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
    pool_output = concatenate([p for p in pool_output])
    normal = BatchNormalization()(pool_output)
    act = Activation("relu")(normal)
    output = Flatten(name = name)(act)
    return output

def gru(input,name):
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Bidirectional(GRU(128,))(x)
    x = BatchNormalization()(x)
    output = Activation('relu',name = name)(x)
    return output