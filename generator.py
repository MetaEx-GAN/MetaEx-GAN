#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:04:31 2017

@author: gama
"""
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Dense,Reshape,concatenate,Flatten,Activation,Permute,multiply
from tensorflow.keras.layers import GRU,Conv1D,GlobalMaxPooling1D,TimeDistributed,RepeatVector,LSTM,MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda,Dropout
from tensorflow.keras.layers import BatchNormalization
import pickle
import numpy as np
import random
from tqdm import tqdm,tqdm_notebook
import json
from sklearn.metrics import f1_score,confusion_matrix,recall_score
import itertools
from functools import reduce
from tqdm.notebook import trange

max_length=20
num_words=5000
embedding_dict=num_words
batch_size=64
start_label=0
MLE_epoch=100
pre_d_epoch=100
adv_step=100


def getG():
    dim=32
    input1 = Input(shape=(max_length,))
    emb=Embedding(num_words+1,dim)
    context=emb(input1)
    LSTM_input = LSTM(dim,return_sequences=True)(context)
    predict = Dense(num_words,activation="softmax")(LSTM_input)
    model = Model(inputs=input1, outputs=predict)
    return model




