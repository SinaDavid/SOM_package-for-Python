# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:05:03 2024

@author: sdd380
"""
import numpy as np
from sklearn.decomposition import PCA
from som_map_functions import *
from som_lininit import *
from som_randinit import *

def som_make(D, *args, **kwargs):
    # Parse input arguments
    # breakpoint()
    if isinstance(D, dict):
        data_name = D['name']
        comp_names = D['comp_names']
        comp_norm = D['comp_norm']
        D = D['data']
    else:
        data_name = 'D'
        pca = PCA(n_components=D.shape[1])
        pca.fit(D)
        comp_names = [f'PC{i+1}' for i in range(D.shape[1])]
        comp_norm = {i: pca.components_[i] for i in range(D.shape[1])}

    dlen, dim = D.shape
    
    # Process additional keyword arguments
    # init = kwargs.get('init', 'lininit')
    # algorithm = kwargs.get('algorithm', 'batch')
    # munits = kwargs.get('munits', None)
    # msize = kwargs.get('msize', None)
    # mapsize = kwargs.get('mapsize', 'normal')
    # lattice = kwargs.get('lattice', 'hexa')
    # shape = kwargs.get('shape', 'sheet')
    # neigh = kwargs.get('neigh', 'gaussian')
    # topol = kwargs.get('topol', None)
    # mask = kwargs.get('mask', None)
    # name = kwargs.get('name', None)
    # tracking = kwargs.get('tracking', 1)
    # training = kwargs.get('training', 'default')
    
    # defaults
    mapsize= ''
    sM = som_map_struct(dim)
    # Determine number of map units if not specified
    if munits is None:
        munits = int(5 * dlen ** 0.54321)

    # Determine map grid size if not specified
    if msize is None:
        # Use PCA to calculate the two biggest eigenvalues
        pca = PCA(n_components=2)
        pca.fit(D)
        eigenvalues = pca.explained_variance_
        ratio = eigenvalues[1] / eigenvalues[0]  # ratio between sidelengths
        # Determine the sidelengths
        msize_x = int(np.sqrt(munits / ratio))
        msize_y = int(np.sqrt(munits * ratio))
        msize = [msize_x, msize_y]
    print("Value of dlen:", dlen)
    # Initialize SOM
    if init == 'lininit':
        # Implement linear initialization
        pass  # Placeholder for linear initialization
    elif init == 'randinit':
        # Implement random initialization
        pass  # Placeholder for random initialization
    else:
        raise ValueError("Invalid value for 'init'. Must be 'lininit' or 'randinit'.")
    
    
    # defaults
    mapsize = ''
    sM = som_map_struct(dim)
    
    sTopol = sM['topol']
    munits = np.prod(sTopol['msize'])  # should be zero
    mask = sM['mask']
    name = sM['name']
    neigh = sM['neigh']
    tracking = 1
    algorithm = 'batch'
    initalg = 'lininit'
    training = 'default'
    
    # args
    i = 0
    while i < len(args):
        arg = args[i]
        argok = 1
        if isinstance(arg, str):
            if arg == 'mask':
                i += 1
                mask = args[i]
            elif arg == 'munits':
                i += 1
                munits = args[i]
            elif arg == 'msize':
                i += 1
                sTopol['msize'] = args[i]
                munits = np.prod(sTopol['msize'])
            elif arg == 'mapsize':
                i += 1
                mapsize = args[i]
            elif arg == 'name':
                i += 1
                name = args[i]
            elif arg == 'comp_names':
                i += 1
                comp_names = args[i]
            elif arg == 'lattice':
                i += 1
                sTopol['lattice'] = args[i]
            elif arg == 'shape':
                i += 1
                sTopol['shape'] = args[i]
            elif arg in ['topol', 'som_topol', 'sTopol']:
                i += 1
                sTopol = args[i]
                munits = np.prod(sTopol['msize'])
            elif arg == 'neigh':
                i += 1
                neigh = args[i]
            elif arg == 'tracking':
                i += 1
                tracking = args[i]
            elif arg == 'algorithm':
                i += 1
                algorithm = args[i]
            elif arg == 'init':
                i += 1
                initalg = args[i]
            elif arg == 'training':
                i += 1
                training = args[i]
            else:
                argok = 0
        elif isinstance(arg, dict) and 'type' in arg:
            if arg['type'] == 'som_topol':
                sTopol = arg
            else:
                argok = 0
        else:
            argok = 0
    
        if not argok:
            print('(som_make) Ignoring invalid argument #', i + 1)
        i += 1
    
    # Map size determination
    
    if bool(sTopol['msize']) or np.prod(sTopol['msize']) == 0:
        if tracking > 0:
            print('Determining map size...')
        if munits==0:
            sTemp = som_topol_struct('dlen', dlen)
            # breakpoint()
            munits = np.prod(sTemp['msize'])
            if mapsize == 'small':
                munits = max(9, np.ceil(munits / 4))
            elif mapsize == 'big':
                munits *= 4
        sTemp= som_topol_struct('data',D, 'munits', munits)    
        sTopol['msize'] = sTemp['msize']
        if tracking > 0:
            print(f"Map size [{sTopol['msize'][0]}, {sTopol['msize'][1]}]")
    
    # breakpoint()
    sMap = som_map_struct(dim, **{'sTopol': sTopol, 'neigh': neigh, 'mask': mask, 'name': name, 'comp_names': comp_names, 'comp_norm': comp_norm})
    
    
    # function
    if algorithm == 'sompak':
        algorithm= 'seq'
        func = 'sompak'
    else:
        func=algorithm
        
    ## initialization
    if tracking > 0:
        print("Initialization...")
        
    if 'initalg' in locals():
        if initalg == 'randinit':
            sMap = som_randinit(D, sMap)
        elif initalg == 'lininit':
            sMap = som_lininit(D,sMap)
    sMap['trainhist'][0] = som_set(sMap['trainhist'][0], ['data_name'], {'data_name': data_name})
    
# Train SOM

    # Return trained SOM struct

