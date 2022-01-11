#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:04:31 2017

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
# Load data
import numpy as np
import random
from tqdm import tqdm,tqdm_notebook
import json
#from src.utils import ruleFormatter 
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



def inference(
    model, 
    inpData = None,
    start_on = 0,
    end_on = max_length,
    num_data = 20000, 
    batch_size = 64,
    que_pad = max_length,
):
    # Initialize
    num_batch = (num_data-1)//batch_size +1
    resp_pred_list = None
    in_batch_list = None
    the_first = True
    idx = np.arange(num_words)
    for b in range(num_batch):
        in_batch = np.zeros((batch_size, que_pad), dtype = int)
        if start_on == 0:
            in_batch[:,0] = start_label#embedding_dict['BOS']
        else:
            in_batch = inpData[b*batch_size:(b+1)*batch_size]
        resp_pred = np.zeros((batch_size, que_pad), dtype = int)
        # Generate the sequence recurrsively.
        for i in range(start_on, end_on):
            # Run
            resp_pred_wv = model.predict(in_batch)
            the_last = resp_pred_wv[:,i]
            the_last = np.array([np.random.choice(idx, p=i) for i in the_last])
            try:
                resp_pred[:,i] = the_last
                in_batch[:,i+1] = the_last
            except:
                resp_pred[:,i] = the_last
        for i in range(len(resp_pred)):
            try:
                index = list(resp_pred[i]).index(6000)#embedding_dict['EOS'])
            except:
                continue
            resp_pred[i,index+1:] = 0
            in_batch[i,index+1:] = 0
        if the_first:
            resp_pred_list = resp_pred
            in_batch_list = in_batch
            the_first = False
        else:
            resp_pred_list = np.vstack((resp_pred_list, resp_pred))
            in_batch_list = np.vstack((in_batch_list, in_batch))        
    resp_pred_list = resp_pred_list[:num_data]
    in_batch_list = in_batch_list[:num_data]
    if start_on != 0:
        resp_pred_list[:,:start_on] = inpData[:,1:start_on+1]
        in_batch_list[:, :start_on+1] = inpData[:,:start_on+1]
    return resp_pred_list, in_batch_list#prediction,decoder input
def model_g_predict_batch(model, _inp_list, batch_size, num_data, step, **kwargs):
    is_first = 1
    y_out = None
    for i in range(0, num_data, batch_size):
        y = model.predict([_inp[i:i+batch_size] for _inp in _inp_list], **kwargs)[:,step]
        if is_first:
            is_first = 0
            y_out = y
        else:
            y_out = np.vstack((y_out, y))
    return y_out

def model_predict_batch(model, _inp_list, batch_size, num_data, **kwargs):
    y_out = np.vstack(
        [ model.predict([_inp[i:i+batch_size] for _inp in _inp_list], **kwargs) 
         for i in range(0, num_data, batch_size)]
    )
    return y_out

def regs_mcmc(model_g, model_d, de_mcmc = None, candidate = 16, start_at = 0, beam = 1):
    # Initialize
    y_mcmc = None
    r_out = None
    que_pad=max_length
    if isinstance(de_mcmc, type(None)):
        de_mcmc = np.zeros((1, que_pad), dtype = int)
        de_mcmc[:,0] = start_label
    y_mcmc = np.zeros((1, que_pad), dtype = int)
    y_mcmc[0,:start_at] = de_mcmc[0, 1:start_at+1]#fix the previous step
    # It determines which word to pass down.
    beam_list = np.ones(que_pad, dtype = int)*beam
    beam_list[:start_at+1] = 1    
    revC = num_words - candidate -1
    # bcList stands for beam-candidate list
    bcList = []
    for i in range(que_pad):
        if i < start_at:
            bcList.append([])
        elif i == start_at:
            bcList.append([revC])
        elif i > start_at:
            bcList.append(list(range(num_words-beam_list[i], num_words)))
        else:
            print('Warning')
    # Generate sequences using MCMC
    for t in range(start_at, que_pad):
        to_expand = beam_list[t]
        the_last = model_g_predict_batch(
            model_g, 
            [de_mcmc], 
            batch_size = 512, 
            num_data = len(de_mcmc), 
            step = t,
        )
        most_possible = np.argsort(the_last, axis = 1)
        most_possible = np.transpose(
            most_possible[:,bcList[t]]).reshape(
            reduce(lambda x,y: x*y, beam_list[:t+1])
        )
        de_mcmc = np.tile(de_mcmc, (to_expand, 1))
        y_mcmc = np.tile(y_mcmc, (to_expand, 1))
        y_mcmc[:,t] = most_possible
        if t+1 < max_length: #10? 
            de_mcmc[:,t+1] = most_possible
    # Rank all synthetic sequences
    de_in = de_mcmc[-1,start_at]
    y_out = y_mcmc[-1,start_at]
    r_mcmc = model_predict_batch(model_d, [y_mcmc], batch_size = 512, num_data = len(y_mcmc))
    r_out = np.mean(r_mcmc)
    # Rank each tokens
    return de_in, y_out, r_out
def regs(model_g, model_d, candidate = 64, beam = 1):
    que_pad=max_length
    de_in = np.zeros((candidate*que_pad, que_pad), dtype = int)
    de_in[:,0] = start_label
    y_out = np.zeros((candidate*que_pad, que_pad), dtype = int)
    r_out = np.zeros((candidate*que_pad, que_pad))
    for q in trange(que_pad):
        if q > 0:
            # Set the previous token by the stochastic one.
            y_stochastic, de_stochastic = inference(model_g, num_data = 1)
            de_in[candidate*q:, :q+1] = de_stochastic[0, :q+1]
            y_out[candidate*q:, :q] = y_stochastic[0, :q]
            r_out[candidate*q:, :q] = 0
            #r_out[candidate*q:, :q] = 1
        for c in range(candidate):
            row = q*candidate + c
            _, y_tmp, r_tmp = regs_mcmc(
                model_g,
                model_d,
                de_in[row].reshape((1,-1)),
                candidate = c,
                start_at = q,
                beam = beam,
            )
            y_out[row,q] = y_tmp
            r_out[row,q] = r_tmp
        # Variance reducing
        r_mean = np.mean(r_out[q*candidate:(q+1)*candidate, q])
        r_out[q*candidate:(q+1)*candidate, q] = r_out[q*candidate:(q+1)*candidate, q] - r_mean
    return de_in, y_out, r_out
def estimate_reward(g_model):
    y_fake_batch, x_fake_batch = inference(           
         g_model, 
         num_data = 128, 
         batch_size = 64,
     )
    G_reward=d_model.predict(np.array(y_fake_batch),batch_size=32)
    G_reward=np.mean(G_reward,axis=0)[0]
    print(G_reward)
    return G_reward


