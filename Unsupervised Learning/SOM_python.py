# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:36:32 2024

@author: sdd380
"""

import os
os.chdir('C:/Users/sdd380/Downloads/ISBS2024_ML-main/ISBS2024_ML-main/Unsupervised Learning/')
#from matplotlib import pyplot as plt
# from sklearn.datasets import load_iris
from som_data_struct import som_data_struct
from som_normalize import som_normalize
from som_make import som_make
from som_bmus import som_bmus
from som_ind2sub import som_ind2sub
import numpy as np
from som_denormalize import som_denormalize
import csv
from reader import *

# iris = load_iris()
# data= iris.data

## create dataset
filename= 'running.csv'


# Read headers
headers = read_headers(filename)

# Read data
data = read_data(filename)


sData = som_data_struct(data.copy())

plotdata = sData['data'].copy()

sData_norm = som_normalize(sData, 'var')

sMap = som_make(sData_norm, *['lattice', 'shape'],**{'lattice':'hexa', 'shape':'sheet'})

Traj_train, Qerrs_train = som_bmus(sMap, sData_norm, 'all')

Traj_train_coord = som_ind2sub(sMap, Traj_train[:,0])
Traj_train_coord = np.concatenate((Traj_train_coord, Qerrs_train[:, [0]]), axis=1)
line1 = np.concatenate((sMap['topol']['msize'], [0]))

M = som_denormalize(sMap['codebook'], *[sMap])

Traj_test, Qerrs_test = som_bmus(M, plotdata, 'all')

Traj_test_coord = som_ind2sub(sMap, Traj_test[:,0])
Traj_test_coord = np.concatenate((Traj_test_coord, Qerrs_test[:, [0]]), axis=1)


## find the lines that hit each neuron
index = [[None for _ in range(line1[1])] for _ in range(line1[0])]

# Iterate over t and q using nested loops
for t in range(0, line1[1]):
    for q in range(0, line1[0]):
        index[q][t] = np.where((Traj_train_coord[:, 0] == q) & (Traj_train_coord[:, 1] == t))[0]



# Flatten index using list comprehension
flattened_index = [item for sublist in index for item in sublist]

# som_show and som_show_add
# som_hits