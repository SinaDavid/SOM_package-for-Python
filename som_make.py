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
        
        argok = 1
        if isinstance(args[i], str):
            if args[i] == 'mask':
                mask = args[i]
                i=i+1
            elif args[i] == 'munits':
                munits = args[i]
                i=i+1
            elif args[i] == 'msize':
                sTopol['msize'] = args[i]
                munits = np.prod(sTopol['msize'])
                i=i+1
            elif args[i] == 'mapsize':
                mapsize = args[i]
                i=i+1
            elif args[i] == 'name':
                name = args[i]
                i=i+1
            elif args[i] == 'comp_names':
                comp_names = args[i]
                i=i+1
            elif args[i] == 'lattice':
                sTopol['lattice'] = args[i]
                i=i+1
            elif args[i] == 'shape':
                sTopol['shape'] = args[i]
                i=i+1
            elif args[i] in ['topol', 'som_topol', 'sTopol']:
                sTopol = args[i]
                munits = np.prod(sTopol['msize'])
                i=i+1
            elif args[i] == 'neigh':
                neigh = args[i]
                i=i+1
            elif args[i] == 'tracking':
                tracking = args[i]
                i=i+1
            elif args[i] == 'algorithm':
                algorithm = args[i]
                i=i+1
            elif args[i] == 'init':
                initalg = args[i]
                i=i+1
            elif args[i] == 'training':
                training = args[i]
                i=i+1
            elif args[i] in ['hexa', 'rect']:
                sTopol['lattice']= args[i]
            elif args[i] in ['sheet', 'cyl', 'toroid']:
                sTopol['shape'] = args[i]
            elif args[i] in ['gaussian','cutgauss','ep','bubble']:
                neigh = args[i]
            elif args[i] in ['seq','batch','sompak']:
                algorithm= args[i]
            elif args[i] in ['small','normal','big']:
                mapsize = args[i]
            elif args[i] in ['randinit','lininit']:
                initalg = args[i]
            elif args[i] in ['short','default','long']:
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
            sTemp = som_topol_struct(*['dlen'], **{'dlen': dlen})
            munits = np.prod(sTemp['msize'])
            if mapsize == 'small':
                munits = max(9, np.ceil(munits / 4))
            elif mapsize == 'big':
                munits *= 4
        sTemp= som_topol_struct(*['data', 'munits'],**{'data':D, 'munits': munits})    
        sTopol['msize'] = sTemp['msize']
        if tracking > 0:
            print(f"Map size [{sTopol['msize'][0]}, {sTopol['msize'][1]}]")
    
    # breakpoint()
    sMap = som_map_struct(dim, *['sTopol','neigh','mask', 'name', 'comp_names', 'comp_norm'],
                                 **{'sTopol': sTopol, 'neigh': neigh, 'mask': mask, 'name': name, 'comp_names': comp_names, 'comp_norm': comp_norm})
    
    
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

