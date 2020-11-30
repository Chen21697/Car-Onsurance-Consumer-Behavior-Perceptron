#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:46:48 2020

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

def gramMatrix(x, y, p):
    return (1+(x@y.T))**p
#%%
lenFeature = len(trainSet_x.columns)
sampleNum = len(trainSet_x)
validNum = len(valSet_x)

x_train = trainSet_x.to_numpy() # transform all data in to numpy form
x_val = valSet_x.to_numpy() # transform all validation set in to numpy form

y_head_train = trainSet_y.to_numpy().reshape(sampleNum) 
y_head_val = valSet_y.to_numpy().reshape(validNum)
#%%
print("Current p value is 1, change the parameter if you wish to")
kernel_t = gramMatrix(x_train, x_train, 1)
kernel_v = gramMatrix(x_train, x_val, 1)
#%%
print("training start")
alpha = np.zeros(sampleNum)
epoch = 100
t_acc = []
v_acc = []

for i in range(epoch):
    for j in range(sampleNum):
        # compute the predicted value
        uj = (y_head_train*alpha)@kernel_t[:,j]
        
        # update alpha
        comp = np.sign(uj*y_head_train[j])
        if comp <= 0:
            alpha[j] += 1
    
    t_pred = (y_head_train*alpha)@kernel_t
    v_pred = (y_head_train*alpha)@kernel_v
    
    epoch_t_acc = accuracy(np.sign(t_pred),y_head_train)
    epoch_v_acc = accuracy(np.sign(v_pred),y_head_val)
    
    t_acc.append(epoch_t_acc)
    v_acc.append(epoch_v_acc)
    
    # accuracy for each epoch
    if i%10 == 0:
        print('epoch:', i, 't_acc:', epoch_t_acc, 'v_acc:', epoch_v_acc)
        
