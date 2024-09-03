# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:36:32 2024

@author: sdd380
"""

import os
os.chdir('C:/Users/sdd380/surfdrive - David, S. (Sina)@surfdrive.surf.nl/Projects/SOM_Workshop/SOM-Workshop/')
#from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from som_data_struct import som_data_struct
from som_normalize import som_normalize
from som_make import som_make
from som_bmus import som_bmus
from som_ind2sub import som_ind2sub
import numpy as np
from som_denormalize import som_denormalize
from som_show import som_show
import copy
import pandas as pd
import csv
from reader import *

## create dataset
filename= 'walking.csv'


# Read headers
# headers = read_headers(filename)

# Read data
data = read_data(filename)


# iris = load_iris()
# data= iris.data

# sData = som_data_struct(data, *['comp_names'],**{'comp_names': headers})
sData = som_data_struct(data)

plotdata = sData['data'].copy()

sData_copy=copy.deepcopy(sData)
sData_norm = som_normalize(sData_copy, 'var')

sData_norm_copy= copy.deepcopy(sData_norm)
sMap = som_make(sData_norm_copy, *['lattice', 'shape'],**{'lattice':'hexa', 'shape':'sheet'})

sMap_copy = copy.deepcopy(sMap)
Traj_train, Qerrs_train = som_bmus(sMap_copy, sData_norm_copy, 'all')

Traj_train_coord = som_ind2sub(sMap_copy, Traj_train[:,0])
Traj_train_coord = np.concatenate((Traj_train_coord, Qerrs_train[:, [0]]), axis=1)
line1 = np.concatenate((sMap['topol']['msize'], [0]))


M = som_denormalize(sMap_copy['codebook'], *[sMap_copy])

h =som_show(sMap)

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