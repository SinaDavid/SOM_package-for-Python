# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:22:24 2024

@author: sdd380
"""
import csv
import numpy as np

def read_headers(filename, delimiter=','):
    with open(filename, 'r') as f:
        headers = f.readline().strip().split(delimiter)
    return headers

def read_data(filename, delimiter=','):
    return np.loadtxt(filename, delimiter=delimiter, skiprows=1)
