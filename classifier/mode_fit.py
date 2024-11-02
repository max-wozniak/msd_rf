# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:04:56 2024

@author: Tyler
"""

SAMPLES_PER_SPECTRUM = 512
AVERAGE_ORDER = 12

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import os


infile = './train/channel1_d_pl.out'
raw_gain = np.genfromtxt(infile, dtype=float)[:, 1]
bad_indices = raw_gain < 0.0
raw_gain[bad_indices] = 0

raw_gain = raw_gain.reshape( (SAMPLES_PER_SPECTRUM, -1) ).T
average_filter = np.ones((AVERAGE_ORDER,), dtype=float)/AVERAGE_ORDER

avg_gain = np.zeros((raw_gain.shape[0], raw_gain.shape[1] - average_filter.shape[0] + 1))


for i in range(raw_gain.shape[0]):
    avg_gain[i, :] = np.convolve(raw_gain[i, :], average_filter, mode='valid')

dG = avg_gain[1:, :] - avg_gain[0:avg_gain.shape[0]-1, :]
avg_gain = avg_gain[1:, :]

plt.plot(avg_gain[80,:])
plt.plot(dG[80, :])
plt.show()

model = svm.LinearSVC()


