# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:04:56 2024

@author: Tyler
"""

SAMPLES_PER_SPECTRUM = 512
AVERAGE_ORDER = 12
NUM_FEATURES = 4
TEST_INDEX = -1

BW_THRESH = 23.0

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

train_path = './train'
infiles = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f)) and '.out' in f]

train_data = np.array([[]])
test_data = np.array([[]])
train_labels = np.array([])
test_labels = np.array([])

for infile in infiles:
    raw_gain = np.genfromtxt(infile, dtype=float)[:, 1]
    bad_indices = raw_gain < 0.0
    raw_gain[bad_indices] = 0
    
    raw_gain = raw_gain.reshape( (SAMPLES_PER_SPECTRUM, -1) ).T
    average_filter = np.ones((AVERAGE_ORDER,), dtype=float)/AVERAGE_ORDER
    
    avg_gain = np.zeros((raw_gain.shape[0], raw_gain.shape[1] - average_filter.shape[0] + 1))
    
    
    for i in range(raw_gain.shape[0]):
        avg_gain[i, :] = np.convolve(raw_gain[i, :], average_filter, mode='valid')
    
    dG = np.abs(avg_gain[1:, :] - avg_gain[0:avg_gain.shape[0]-1, :])
    avg_gain = avg_gain[1:, :]
    
    train_data_tmp = np.zeros((avg_gain.shape[0], NUM_FEATURES))
        
    fmin = 0
    fmax = avg_gain.shape[1]-1    
    
    for i in range(avg_gain.shape[0]):
        if avg_gain[i][0] < BW_THRESH:
            for j in range(avg_gain.shape[1]-1):
                if avg_gain[i][j] < BW_THRESH and avg_gain[i][j+1] > BW_THRESH:
                    fmin = j
                    break
        if avg_gain[i][-1] < BW_THRESH:
            for j in range(avg_gain.shape[1]-1, 0, -1):
                if avg_gain[i][j] < BW_THRESH and avg_gain[i][j-1] > BW_THRESH:
                    fmax = j
                    break
        train_data_tmp[i, 0] = fmax-fmin
        train_data_tmp[i, 1] = np.var(avg_gain[i, fmin+20:fmax+1-20])
        train_data_tmp[i, 2] = np.mean(avg_gain[i, fmin+20:fmax+1-20])
        train_data_tmp[i, 3] = np.max(dG[i, :])
        
    label = 1 if 'fm.out' in infile else 0
    labels = np.ones((train_data_tmp.shape[0], 1), dtype=int)*label
    
    xtrain, xtest, ytrain, ytest = train_test_split(train_data_tmp, labels, test_size=0.15)
    
    if train_data.shape[1] == 0:
        train_data = xtrain
        test_data = xtest
    else:
        train_data = np.row_stack((train_data, xtrain))
        test_data = np.row_stack((test_data, xtest))
    
    if train_labels.shape[0] == 0:
        train_labels = ytrain
        test_labels = ytest
    else:
        train_labels = np.concatenate((train_labels, ytrain))
        test_labels = np.concatenate((test_labels, ytest))
    
            
'''
print('Fmin: ', fmin)
print('Fmax: ', fmax)
print('Width: ', fmax-fmin)
print('Signal Std: ', np.var(avg_gain[-1, fmin+20:fmax+1-20]))
print('Signal Avg: ', np.mean(avg_gain[-1, fmin+20:fmax+1-20]))
'''


plt.plot(avg_gain[TEST_INDEX,:])
plt.plot(dG[TEST_INDEX, :])
plt.show()

print(test_labels.shape, train_labels.shape)

model = svm.LinearSVC(dual=False, max_iter=10000)
model.fit(train_data, train_labels)
print(model.score(test_data, test_labels))
print(model.coef_, model.intercept_)

