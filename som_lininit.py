# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:25:45 2024

@author: sdd380
"""

import numpy as np
from som_map_functions import *
from som_side_functions import *
from som_train_struct import *

def som_lininit(D, *args, **kwargs):
    
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
       if isinstance(args[i-1], str):
           
           if args[i-1]== 'munits':
              munits= args[i-1]
              sTopol['msize']=0
              i = i+1
           elif args[i-1] == 'msize':
               sTopol['msize']= args[i-1]
               munits = np.prod(sTopo['msize'])
               i= i+1
           elif args[i-1]== 'lattice':
               sTopol['lattice']= args[i-1]
               i=i+1
           elif args[i-1]=='shape':
               sTopol['shape'] =args[i-1]
               i=i+1
           elif args[i-1]==['som_topol','sTopol', 'topol']:
               sTopol= args[i-1]
               i=i+1
           elif args[i-1]==['som_map','sMap','map']:
               sMap = args[i-1]
               sTopo = sMap['topol']
               i=i+1
           elif args[i-1]==['hexa','rect']:
                  sTopol['lattice']= args[i-1]
           elif args[i-1]==['sheet','cyl','toroid']:
               sTopol['shape']= args[i-1]
           else:
               argok=0
       elif isinstance(args[i-1], dict) and 'type'in args[i-1]: 
           if args[i-1]['type']=='som_topol':
               sTopol = args[i-1]
           elif args[i-1]['type']== 'som_map':
               sMap = args[i-1]
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
    
    if structmode ==1:
        sMap = som_set(sMap, *['comp_names', 'comp_norm'], **{'comp_names': comp_names, 'comp_norm': comp_norm})
    
    breakpoint()
    ## initialization
    # train struct
    sTrain = som_train_struct(*['algorithm'],**{'algorithm':'lininit'})
    sTrain = som_set(sTrain, *['data_name'], **{'data_name': data_name})
    
    msize = sMap['topol']['msize']
    mdim = len(msize)
    munits = np.prod(msize)
    
    dlen, dim = D.shape
    if dlen<2:
        raise ValueError('Linear map initialization requires at least two NaN-free samples.')
        return
    
    
   
    ## compute principle components
    if dim > 1 and np.sum(msize > 1) > 1:
        # Calculate mdim largest eigenvalues and their corresponding eigenvectors
        # Autocorrelation matrix
        A = np.zeros((dim, dim))
        me = np.zeros(dim)
        
        for i in range(dim):
            me[i] = np.nanmean(D[:, i])  # Mean of finite values
            D[:, i] = D[:, i] - me[i]
        
        for i in range(dim):
            for j in range(i, dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = np.sum(c) / len(c)
                A[j, i] = A[i, j]
        
        # Take mdim first eigenvectors with the greatest eigenvalues
        eigvals, eigvecs = np.linalg.eig(A)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        eigvecs = eigvecs[:, :mdim]
        eigvals = eigvals[:mdim]
        
        # Normalize eigenvectors to unit length and multiply them by corresponding (square-root-of-)eigenvalues
        for i in range(mdim):
            eigvecs[:, i] = (eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])) * np.sqrt(eigvals[i])
    
    else:
        me = np.zeros(dim)
        V = np.zeros(dim)
        
        for i in range(dim):
            inds = np.where(~np.isnan(D[:, i]))[0]
            me[i] = np.nanmean(D[inds, i])
            V[i] = np.nanstd(D[inds, i])
    
    ## initialize codebook vectors
        
    return sMap