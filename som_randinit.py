# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:47:54 2024

@author: sdd380
"""
import numpy as np
from som_topol_struct import som_topol_struct 
from som_map_struct import som_map_struct
from som_set import som_set
from som_train_struct import som_train_struct
from som_unit_coords import som_unit_coords
from datetime import datetime


def som_randinit(D, *args, **kwargs):
    
    
############# DETAILED DESCRIPTION ########################################
#
# som_randinit
#
# PURPOSE
#
# Initializes a SOM with random values.
#
# SYNTAX
#
#  sMap = som_randinit(D)
#  sMap = som_randinit(D,sMap);
#  sMap = som_randinit(D,'munits',100,'hexa');
#
# DESCRIPTION
#
# Initializes a SOM with random values. If necessary, a map struct
# is created first. For each component (xi), the values are uniformly
# distributed in the range of [min(xi) max(xi)]. 
#
# REQUIRED INPUT ARGUMENTS
#
#  D                 The training data.
#           (struct) Data struct. If this is given, its '.comp_names' and 
#                    '.comp_norm' fields are copied to the map struct.
#           (matrix) data matrix, size dlen x dim
#  
# OPTIONAL INPUT ARGUMENTS 
#
#  argID (string) Argument identifier string (see below).
#  value (varies) Value for the argument (see below).
#
#  The optional arguments can be given as 'argID',value -pairs. If an
#  argument is given value multiple times, the last one is used. 
#
#  Here are the valid argument IDs and corresponding values. The values 
#  which are unambiguous (marked with '*') can be given without the 
#  preceeding argID.
#  'dlen'         (scalar) length of the training data
#  'data'         (matrix) the training data
#                *(struct) the training data
#  'munits'       (scalar) number of map units
#  'msize'        (vector) map size
#  'lattice'     *(string) map lattice: 'hexa' or 'rect'
#  'shape'       *(string) map shape: 'sheet', 'cyl' or 'toroid'
#  'topol'       *(struct) topology struct
#  'som_topol','sTopol'    = 'topol'
#  'map'         *(struct) map struct
#  'som_map','sMap'        = 'map'
#
# OUTPUT ARGUMENTS
# 
#  sMap     (struct) The initialized map struct.
    if isinstance(D, dict):
        data_name = D['name']
        comp_names = D['comp_names']
        comp_norm = D['comp_norm']
        D= D['data']
        struct_mode =1
    else:
        data_name = 'D'
        structmode =0
        
    dlen, dim = D.shape  
    
    sMap = []
    sTopol = som_topol_struct()
    sTopol['msize']=0
    munits= None
    
    i=0
    while i <= len(args)-1:
       argok=1
       if isinstance(args[i], str):
           
           if args[i]== 'munits':
              munits= kwargs[args[i]]
              sTopol['msize']=0
              
           elif args[i] == 'msize':
               sTopol['msize']= kwargs[args[i]]
               munits = np.prod(sTopol['msize'])
               
           elif args[i]== 'lattice':
               sTopol['lattice']= kwargs[args[i]]
               
           elif args[i]=='shape':
               sTopol['shape'] = kwargs[args[i]]
               
           elif args[i] in ['som_topol','sTopol', 'topol']:
               sTopol= kwargs[args[i]]
               
           elif args[i] in ['som_map','sMap','map']:
               sMap = kwargs[args[i]]
               sTopol = sMap['topol']
               
           elif args[i] in ['hexa','rect']:
                  sTopol['lattice']= kwargs[args[i]]
           elif args[i] in ['sheet','cyl','toroid']:
               sTopol['shape']= kwargs[args[i]]
           else:
               argok=0
       elif isinstance(args[i], dict) and 'type'in args[i]: 
           if args[i]['type']=='som_topol':
               sTopol = kwargs[args[i]]
           elif args[i]['type']== 'som_map':
               sMap = kwargs[args[i]]
               sTopol= sMap['topol']
           else:
               argok=0
       else:
           argok=0
            
       if argok ==0:
           print('(som_topol_struct) Ignoring invalid argument #' + str(i))
       i = i+1
    
    if len(sTopol['msize'])==1:
        sTopol["msize"].append(1)
       
    if bool(sMap):
        munits, dim2 = sMap['codebook'].shape
    
    if dim2 != dim:
        raise ValueError('Map and data must have the same dimension.')
    
    # create map
    # map struct
    if bool(sMap):
        sMap = som_set(sMap, *['topol'], **{'topol': sTopol})
    else:
        if not np.prod(sTopol['msize']):
            if np.isnan(munits):
                sTopol = som_topol_struct(*['data', 'sTopol'], **{'data': D, 'sTopol': sTopol})
            else:
                sTopol = som_topol_struct(*['data', 'munits', 'sTopol'], **{'data': D,'munits': munits, 'sTopol': sTopol})
            
            sMap = som_map_struct(dim, args, kwargs)
            
    if structmode ==1:
        sMap = som_set(sMap, *['comp_names', 'comp_norm'], **{'comp_names': comp_names, 'comp_norm': comp_norm})
    
    ## initialization
    # train struct
    sTrain = som_train_struct(*['algorithm'],**{'algorithm':'randinit'})
    sTrain = som_set(sTrain, *['data_name'], **{'data_name': data_name})
    
    munits = np.prod(sMap['topol']['msize'])
    sMap['codebook']= np.random.rand(munits, dim)
    
    # set interval of each component to correct value
    for i in range(dim):
        inds = np.where(~np.isnan(D[:, i]))[0] and ~np.isinf(D[:, i])[0]
        if ~bool(inds):
            mi = 0
            ma = 1
        else:
            ma = max(D[inds,i])
            mi = min(D[inds,i])
        sMap['codebook'][:,i] = (ma - mi) * sMap['codebook'][:,i] + mi
    
    # training struct
    sTrain = som_set(sTrain, *['time'], **{'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    sMap['trainhist'] = sTrain.copy()
        
        
        
    return sMap