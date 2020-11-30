#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:42:04 2020

@author: yuwenchen
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
trainSet_x = pd.DataFrame(pd.read_csv('pa3_train_X.csv'))
trainSet_y = pd.DataFrame(pd.read_csv('pa3_train_y.csv'))

valSet_x = pd.DataFrame(pd.read_csv('pa3_dev_X.csv'))
valSet_y = pd.DataFrame(pd.read_csv('pa3_dev_y.csv'))
#%%
def accuracy(predict, ans):
    length = len(ans)
    
    compList = predict == ans
    # the number of correct
    correct = sum(compList)
    return correct/length
#%%
lenFeature = len(trainSet_x.columns)
sampleNum = len(trainSet_x)
validNum = len(valSet_x)

x_train = trainSet_x.to_numpy() # transform all data in to numpy form
x_val = valSet_x.to_numpy() # transform all validation set in to numpy form

y_head_train = trainSet_y.to_numpy().reshape(sampleNum) 
y_head_val = valSet_y.to_numpy().reshape(validNum)
#%%
w_onl = np.zeros(lenFeature)
w_avg = np.zeros(lenFeature)
count = 1
epoch = 100
t_acc_o = []
v_acc_o = []

for i in range(epoch):
    for j in range(sampleNum):
        if y_head_train[j]*(w_onl.T@x_train[j]) <= 0:
            w_onl = w_onl + y_head_train[j]*x_train[j]

        w_avg = (count*w_avg + w_onl)/(count+1)
        count += 1
    
    # for online perciptron
    #t_pred = w_onl@x_train.T
    #v_pred = w_onl@x_val.T
    
    # for average perceptron
    t_pred = w_avg@x_train.T
    v_pred = w_avg@x_val.T
    
    epoch_t_acc = accuracy(np.sign(t_pred),y_head_train)
    epoch_v_acc = accuracy(np.sign(v_pred),y_head_val)
    
    t_acc_o.append(epoch_t_acc)
    v_acc_o.append(epoch_v_acc)
    
    
    # accuracy for each epoch
    if i%10 == 0:
        print('epoch:', i, 't_acc:', epoch_t_acc, 'v_acc:', epoch_v_acc)