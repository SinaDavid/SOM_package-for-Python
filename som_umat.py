# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:29:58 2024

@author: sdd380
"""
import numpy as np
from som_set import som_set
import math

def som_umat(sMap, *args, **kwargs):
    if isinstance(sMap, dict):
        M= sMap['codebook']
        sTopol = sMap['topol']
        mask = sMap['mask']
    elif isinstance(sMap, np.ndarray):
        M = sMap
        si = M.shape
        dim = si[-1]
        if len(si)>2:
            msize= si[:-1]
        else:
            msize = [si[0],1]
        munits = np.prod(msize)
        sTopol = som_set('som_topol', *['msize', 'lattice', 'shape'], **{'msize': msize, 'lattice': 'rect', 'shape': 'sheet'}) 
        mask = np.ones((dim,1))
    
    mode = 'median'
    i=0
    while i <= len(args)-1:
        argok=1
        if isinstance(args[i],str):
            if args[i]=='mask':
                mask= kwargs[args[i]]
            elif args[i]=='msize':
                sTopol['msize']= kwargs[args[i]]
            elif args[i] in {'topol','som_topol','sTopol'}:
                sTopol=kwargs[args[i]]
            elif args[i]=='mode':
                mode= kwargs[args[i]]
            elif args[i] in {'hexa','rect'}:
                sTopol['lattice']= args[i]
            elif args[i] in {'min','mean','median','max'}:
                mode = args[i]
            
            elif 'type' in kwargs[args[i]]:
                if kwargs[args[i]]['type']=='som_topol':
                    sTopol = kwargs[args[i]]
                elif kwargs[args[i]]['type']=='som_map':
                    sTopol = kwargs[args[i]]['topol']
            else:
                argok=0
        elif isinstance(args[i], dict) and 'type' in args[i]:
            if args[i]['type']=='som_topol':
                sTopol= args[i]
            elif args[i]['type']=='som_map':
                sTopol= args[i]['topol']
            else:
                argok=0
        else:
            argok=0
        if argok==0:
            print("(som_umat) Ignoring invalid argument #{i}")
        i=i+1
    munits, dim = M.shape
    if np.prod(sTopol['msize']) != munits:
        raise ValueError('Map grid size does not match the number of map units.')
    if len(sTopol['msize'])>2:
        raise ValueError('Can only handle 1- and 2-dimensional map grids.')
    if np.prod(sTopol['msize'])==1:
        print("Only one codebook vector.")
        U = []
        return
    if sTopol['shape']=='sheet':
        print("The ' sTopol.shape ' shape of the map ignored. Using sheet instead.")
    
    ## initialize variables
    y = sTopol['msize'][0]
    x = sTopol['msize'][1]
    lattice = sTopol['lattice']
    shape = sTopol['shape']
    M_reshaped = np.empty((dim, y, x))

    # Fill M_reshaped column by column
    for col in range(dim):
        for slice_idx in range(x):
            start_row_idx = slice_idx * y
            end_row_idx = start_row_idx + y
            M_reshaped[col, :, slice_idx] = M[start_row_idx:end_row_idx, col]                                          
    M=M_reshaped.copy()
    
    ux = 2 * x -1
    uy = 2 * y -1
    U = np.zeros((uy,ux))
    calc = f'{mode}(a)'
    
    if mask.shape[1] >1:
        mask = mask.T
    
    ## u-matrix computation
    # if lattice == 'rect':
    #     for j in range(0,y):
    #         for i in range(0,x):
    #             if i<x:
    #                 dx = (M[:,j,i] - M[:,j, i+1])**2
    #                 U[2*j-1,2*i]= math.sqrt(mask.T*dx[:])
    #             if j<y:
    #                 dy = (M[:,j,i] - M[:,j+1,i])**2
    #                 U[2*j,2*i-1]= math.sqrt(mask.T*dy[:])
    #             if j<y and i<x:
    #                 dz1 = (M[:,j,i]- M[:,j+1,i+1])**2
    #                 dz2 = (M[:,j+1,i]- M[:,j,i+1])**2
    #                 U[2*j,2*i]= (math.sqrt(mask.T*dz1[:])+math.sqrt(mask.T*dz2[:]))/(2* math.sqrt(2))
    # elif lattice == 'hexa':
    #     for j in range(0,y):
    #         for i in range(0,x):
    #             if i<x:
                    # dx[ = np.reshape(((M[:,j,i] - M[:,j, i+1])**2),(dim,1,1))
                    # U[2*j, 2*i-1]= math.sqrt(mask.T*dx[:])
                    
        
 