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


def getD():
    dim=64
    resp = Input((max_length,))
    embedding =Embedding(num_words,
                         dim,
                         mask_zero=False,
                         input_length=(max_length),
                         trainable = True)
    resp_emb = embedding(resp)
    context = Conv1D(dim*2,5)(resp_emb)
    context=BatchNormalization()(context)
    context = GlobalMaxPooling1D()(context)
    #context= LSTM(wv_dim)(resp_emb)
    #context= LayerNormalization()(context)
    scalar = Dense(1, activation = 'sigmoid')(context)
    model = Model(
        resp,
        scalar
    )
    return model


