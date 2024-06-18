# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:25:45 2024

@author: sdd380
"""

import numpy as np
from som_map_functions import *
from som_side_functions import *


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
    breakpoint()
    sMap = []
    sTopol = som_topol_struct()
    sTopol['msize']=0
    munits= None
    
    i=1
    while i <= len(args):
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
    breakpoint()
    if bool(sMap):
        sMap = som_set(sMap, *['topol'], **{'topol': sTopol})
    
    
    breakpoint()
    return sMap