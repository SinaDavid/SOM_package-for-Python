# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:39:10 2024

@author: sdd380
"""
import matplotlib.pyplot as plt
import numpy as np   
from matplotlib.colors import LinearSegmentedColormap
import math
from som_cplane import som_cplane
from som_umat import som_umat

def som_show(sMap, *args):
    ############ DETAILED DESCRIPTION #######################################
    #
    # som_show
    #
    # PURPOSE 
    #
    # Shows basic visualizations of SOM: component planes, unified distance
    # matrices as well as empty planes and fixed color planes.
    
    if isinstance(sMap, dict):
        if sMap['type']== 'som_map':
            ok=1
        else:
            raise ValueError('Map struct is invalid!')
    else:
        raise ValueError('Requires a map struct!')
    
    munits, d = sMap['codebook'].shape
    msize = sMap['topol']['msize']
    lattice= sMap['topol']['lattice']
    
    if len(msize)>2:
        raise ValueError('This visualizes only 2D maps!')
    if len(args) % 2 >0:
        raise ValueError('Mismatch in identifier-value  pairs.')
    
    ## read in optional arguments
    if not bool(args):
        args = {'umat': 'all', 'comp': 'all'}
    
    
    Plane=[]
    Plane.append({'mode': list(args.keys())[0],'value': range(d), 'name': 'U-matrix'})
    for i in range(d):
        Plane.append({'mode': 'comp', 'value': i, 'name': []})
    
    General={}
    General['comp']=list(range(d+1))
    General['size']=[]
    General['scale']=[]
    General['colorbardir']=[]
    General['edgecolor']=[]
    General['footnote']=sMap['name']
    
    plt.rcParams['image.cmap'] = 'viridis'
    gray_custom = plt.get_cmap('gray', 64)  # Get the 'gray' colormap with 64 levels
    gray_custom_colors = gray_custom(np.linspace(0, 1, 64)) ** 0.5  # Apply power transformation
    custom_cmap = LinearSegmentedColormap.from_list('custom_gray', gray_custom_colors)

    General['colormap']=custom_cmap
    General['subplots']=[]
    
    if not bool(Plane):
        args={'sMap': sMap, 'umat': 'all', 'comp': 'all'}
    
    if not bool(General['colorbardir']):
       General['colorbardir']= 'vert'
    
    if not bool(General['scale']):
        General['scale']= 'denormalized'
    
    if not bool(General['size']):
        General['size']=1
    
    if not bool(General['edgecolor']):
        General['edgecolor']= 'none'
    
    ## action
    n = len(Plane)
    # get the unique component indices
    c = [value for value in General['comp'] if value > 0]
    c = sorted(set(c) - {0, -1})
    c = [value for value in c if not (isinstance(value, float) and math.isnan(value))]

    # estimate the suitible dimension for subplots
    if not bool(General['subplots']):
        y = int(np.ceil(math.sqrt(n)))
        x = int(np.ceil(n/y))
    else:
        y = General['subplots'][1]
        x = General['subplots'][0]
        if y * x < n:
            raise ValueError(f"Given subplots grid size is too small: should be >= {n}")
    
    plt.clf()  # Clear the current figure
    fig, h_axes = plt.subplots(nrows=x, ncols=y)  # Create a figure with subplots
    for i in range(0, n):
        subplot_index = (i - 1) // y, (i - 1) % y  # Convert subplot index to (row, col)
        ax = h_axes[subplot_index]
        
        if Plane[i]['mode']=='comp':
            tmp_h = som_cplane(lattice, msize, sMap['codebook'][:,General['comp'][i]], General['size'])
            tmp_h['Edgecolor']=General['edgecolor']
        if Plane[i]['mode']=='umat':
            u = som_umat(sMap['codebook'][:,Plane[i]['value']], *['topol','mode','mask'], 
                         **{'topol': sMap['topol'],'mode': 'median','mask': sMap['mask'][Plane[i]['value']]})
    plt.show()