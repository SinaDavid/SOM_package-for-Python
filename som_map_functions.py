# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:18:03 2024

@author: sdd380
"""

import numpy as np
from som_side_functions import *
from datetime import datetime

def som_map_struct(dim, **kwargs):
    # default values
    
    
    
   
    sTopol = som_set('som_topol', *['lattice', 'shape'], **{'lattice': 'hexa', 'shape': 'sheet'})
    neigh = 'gaussian'
    mask = np.ones((dim, 1))
    name = f'SOM {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'  # Update datetime formatting
    labels = [''] * np.prod(sTopol['msize'])
    comp_names = [f'Variable{i}' for i in range(1, dim + 1)]
    comp_norm = [None] * dim

    # args
    # if args:
    i = 1
    if 'kwargs' in locals() and bool(kwargs):
        items = list(kwargs.keys())
    
        
    
        while i <= len(kwargs):
            argok = 1
            if isinstance(items[i-1], str) and not isinstance(kwargs[items[i-1]],dict):
                if items[i-1] == 'mask':
                    mask = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'msize':
                    sTopol['msize'] = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'labels':
                    labels = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'name':
                    name = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'comp_names':
                    comp_names = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'comp_norm':
                    comp_norm = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'lattice':
                    sTopol['lattice'] = args[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'shape':
                    sTopol['shape'] = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] in ['topol', 'som_topol', 'sTopol']:
                    sTopol = kwargs[items[i-1]]
                    i =i+ 1
                elif items[i-1] == 'neigh':
                    neigh = kwargs[items[i-1]]
                    i =i+ 1
                elif 'hexa' in kwargs[items[i-1]] or 'rect'in kwargs[items[i-1]]:
                    sTopol['lattice'] = kwargs[items[i-1]]
                elif 'sheet' in kwargs[items[i-1]] or 'cyl' in kwargs[items[i-1]] or 'teroid' in kwargs[items[i-1]]:
                    sTopol['shape'] = kwargs[items[i-1]]
                elif 'gaussian' in kwargs[items[i-1]] or 'cutgauss' in kwargs[items[i-1]] or 'ep' in kwargs[items[i-1]] or 'bubble' in kwargs[items[i-1]]:
                    neigh = kwargs[items[i-1]]
                else:
                    argok = 0
            elif isinstance(kwargs[items[i-1]], dict) and 'type' in kwargs[items[i-1]]:
                if kwargs[items[i-1]]['type'] == 'som_topol':
                    sTopol = kwargs[items[i-1]]
                else:
                    argok = 0
            else:
                argok = 0
    
            if not argok:
                print(f'(som_map_struct) Ignoring invalid argument #{i + 1}')
            i =i+ 1

    # create the SOM
    # if sTopol['msize'] == 0:
    # # Initialize codebook as an empty array with dim columns
    #     codebook = np.empty((0, dim))
    # else:
    # Otherwise, generate codebook using random values
    codebook = np.random.rand(np.prod(sTopol['msize']), dim)
    
    sTrain = som_set('som_train', *['time','mask'], **{'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mask':mask})  # Update time formatting
    
    sMap = som_set('som_map', *['codebook', 'topol', 'neigh', 'labels', 'mask', 'comp_names', 'name', 'comp_norm', 'train_hist'], 
        **{'codebook': codebook,'topol': sTopol, 'neigh': neigh, 'labels': labels, 'mask': mask, 
            'comp_names': comp_names, 'name': name, 'comp_norm': comp_norm,'train_hist': sTrain})
    
    return sMap

# som_topol_struct
def som_topol_struct(*args):
    """
    SOM_TOPOL_STRUCT Default values for SOM topology.

    sTopol = som_topol_struct([[argID,] value, ...])

    Input and output arguments ([]'s are optional):
    [argID,  (string) Default map topology depends on a number of 
    value]  (varies) factors (see below). These are given as a 
                        argument ID - argument value pairs, listed below.

    sT       (dict) The ready topology dictionary.

    Topology dictionary contains values for map size, lattice (default is 'hexa')
    and shape (default is 'sheet'). Map size depends on training data and the
    number of map units. The number of map units depends on number of training
    samples.
    """
    # initialize
    # first line in matlab is with som_set
    
    sTopol = som_set('som_topol', *['lattice', 'shape'], **{'lattice': 'hexa', 'shape': 'sheet'})
    # breakpoint()
    D = []
    dlen = np.nan
    dim = 2
    munits = np.nan
    
    
    # args
    i = 0
    while i < len(args):
        argok = 1
        if isinstance(args[i], str):
            if args[i] == 'dlen':
                i += 1
                dlen = args[i]
            elif args[i] == 'munits':
                i += 1
                munits = args[i]
                sTopol['msize'] = 0
            elif args[i] == 'msize':
                i += 1
                sTopol['msize'] = args[i]
            elif args[i] == 'lattice':
                i += 1
                sTopol['lattice'] = args[i]
            elif args[i] == 'shape':
                i += 1
                sTopol['shape'] = args[i]
            elif args[i] == 'data':
                i += 1
                if isinstance(args[i], dict):
                    D = args[i]['data']  
                else:
                    D= args[i]
                dlen, dim = D.shape
            elif args[i] in ['hexa', 'rect']:
                sTopol['lattice'] = args[i]
            elif args[i] in ['sheet', 'cyl', 'toroid']:
                sTopol['shape'] = args[i]
            elif args[i] in ['som_topol', 'sTopol', 'topol']:
                i += 1
                if 'msize' in args[i] and np.prod(args[i]['msize']):
                    sTopol['msize'] = args[i]['msize']
                if 'lattice' in args[i]:
                    sTopol['lattice'] = args[i]['lattice']
                if 'shape' in args[i]:
                    sTopol['shape'] = args[i]['shape']
            else:
                argok = 0
        elif isinstance(args[i], dict) and 'type' in args[i]:
            if args[i]['type'] == 'som_topol':
                if 'msize' in args[i] and np.prod(args[i]['msize']):
                    sTopol['msize'] = args[i]['msize']
                if 'lattice' in args[i]:
                    sTopol['lattice'] = args[i]['lattice']
                if 'shape' in args[i]:
                    sTopol['shape'] = args[i]['shape']
            elif args[i]['type'] == 'som_data':
                D = args[i]['data']
                dlen, dim = D.shape
            else:
                argok = 0
        else:
            argok = 0

        if not argok:
            print(f'(som_topol_struct) Ignoring invalid argument #{i}')
        i += 1
    
    if np.prod(sTopol['msize']) == 0 and bool(sTopol['msize']):
        return sTopol
    
    # otherwise, decide msize
    # first (if necessary) determine the number of map units (munits)
    if np.isnan(munits):
        if not np.isnan(dlen):
            munits = np.ceil(5 * np.sqrt(dlen))
        else:
            munits = 100  # just a convenient value
            
    # then determine the map size (msize)
    # breakpoint()
    if dim == 1:  # 1-D data
        
        sTopol['msize'] = [1, np.ceil(munits)]
    elif len(D)<2:
        sTopol['msize']=[]
        sTopol['msize'].append(round(math.sqrt(munits)))
        sTopol['msize'].append(round(munits / sTopol['msize'][0]))
        print(" sTopol first round done")
    else:
        #determine map size based on eigenvalues
        # initialize xdim/ydim ratio using principal components of the input
        # space; the ratio is the square root of ratio of two largest eigenvalues	
        # autocorrelation matrix
        A = np.full((dim, dim), np.inf)
        for i in range(dim):
            column = D[:, i]
            valid_values = column[np.isfinite(column)]
            mean_val = np.mean(valid_values)
            D[:, i] -= mean_val
        for i in range(dim):
            for j in range(i, dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = np.sum(c) / len(c)
                A[j, i] = A[i, j]
        eigvals, eigvecs = np.linalg.eig(A)

        # Sort the eigenvalues in ascending order
        sorted_indices = np.argsort(eigvals)
        sorted_eigvals = eigvals[sorted_indices]

        # Return the sorted eigenvalues
        eigvals = sorted_eigvals   
        
        if eigvals[-1] == 0 or eigvals[-2] * munits < eigvals[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigvals[-1] / eigvals[-2])

        if sTopol['lattice'] == 'hexa':
            # breakpoint()
            sTopol['msize']=[None, None]
            sTopol['msize'][1] = min(munits, round(np.sqrt(munits / ratio * np.sqrt(0.75))))
        else:
            sTopol['msize'][1] = min(munits, round(np.sqrt(munits / ratio)))
        sTopol['msize'][0]= round(munits/sTopol['msize'][1]) 
        if min(sTopol['msize'])==1:
            sTopol['msize']=(1, max(sTopol['msize']))
            
    return sTopol



