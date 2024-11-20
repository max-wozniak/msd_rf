import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

SAMP_NUM = 50

infile = '.\\train\\channel1_ant.out'
raw_gain = np.genfromtxt(infile, dtype=float)[:, 1]
raw_gain = raw_gain[SAMP_NUM*512:(SAMP_NUM+1)*512]

plt.plot(raw_gain)
plt.show()

