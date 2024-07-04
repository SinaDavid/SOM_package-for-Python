# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:02:59 2024

@author: sdd380
"""
import matplotlib.pyplot as plt
import numpy as np   
from matplotlib.colors import LinearSegmentedColormap


def check_args(args, munits, dim, name):
    
    Plane = ([])
    General={}
    General['comp']=[]
    General['size']=[]
    General['scale']=[]
    General['colorbardir']=[]
    General['edgecolor']=[]
    General['footnote']=name
    
    plt.rcParams['image.cmap'] = 'viridis'
    gray_custom = plt.get_cmap('gray', 64)  # Get the 'gray' colormap with 64 levels
    gray_custom_colors = gray_custom(np.linspace(0, 1, 64)) ** 0.5  # Apply power transformation
    custom_cmap = LinearSegmentedColormap.from_list('custom_gray', gray_custom_colors)

    General['colormap']=custom_cmap
    General['subplots']=[]
    
    for i in range(len(args)):
        if not isinstance(list(args.keys())[i], str):
            raise ValueError('Invalid input identifier names or input argument order.')
    
        if list(args.keys())[i] in {'comp', 'compi'}:
            if not list(args.values())[i]:
                args.values()
                
    return Plane, General