# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:36:32 2024

@author: sdd380
"""

import os
os.chdir('C:/Users/sdd380/surfdrive - David, S. (Sina)@surfdrive.surf.nl/Projects/SOM_Workshop/')
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from som_load import *
from som_normalize import som_normalize
from som_side_functions import *
from som_make import *



iris = load_iris()
data= iris.data

sData = som_data_struct(data)

plotdata = sData['data']

sData_norm = som_normalize(sData, 'var')

sMap = som_make(sData_norm, 'hexa', 'sheet')

## in som_map_functions line 97 comparing to matlab!

# weiter mit som_make