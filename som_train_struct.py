# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:33:54 2024

@author: sdd380
"""
import numpy as np
from som_side_functions import *

def som_train_struct(*args, **kwargs):
    
    #SOM_TRAIN_STRUCT Default values for SOM training parameters.
    #
    # sT = som_train_struct([[argID,] value, ...])
    #
    #  sTrain = som_train_struct('train',sM,sD);
    #  sTrain = som_train_struct('finetune','data',D); 
    #  sTrain = som_train_struct('previous',sT0);
    # 
    #  Input and output arguments ([]'s are optional): 
    #    [argID,  (string) Several default values depend on other SOM parameters
    #     value]  (varies) or on the proporties of a data set. See below for a
    #                      a list of required and optional arguments for
    #                      different parameters, and well as the list of valid 
    #                      argIDs and associated values. The values which are 
    #                      unambiguous can be given without the preceeding argID.
    #
    #    sT       (struct) The training struct.
    #
    # Training struct contains values for training and initialization
    # parameters. These parameters depend on the number of training samples,
    # phase of training, the training algorithm.
    # 
    # Here are the valid argument IDs and corresponding values. The values which
    # are unambiguous (marked with '*') can be given without the preceeding rgID.
    #  'dim'          (scalar) input space dimension
    #  'dlen'         (scalar) length of the training data
    #  'data'         (matrix / struct) the training data
    #  'munits'       (scalar) number of map units
    #  'msize'        (vector) map size
    #  'previous'     (struct) previous training struct can be given in 
    #                          conjunction with 'finetune' phase (see below) 
    #  'phase'       *(string) training phase: 'init', 'train', 'rough' or 'finetune'
    #  'algorithm'   *(string) algorithm to use: 'lininit', 'randinit', 'batch' or 'seq'
    #  'map'         *(struct) If a map struct is given, the last training struct
    #                          in '.trainhist' field is used as the previous training
    #                          struct. The map size and input space dimension are 
    #                          extracted from the map struct.
    #  'sTrain'      *(struct) a train struct, the empty fields of which are
    #                          filled with sensible values
    #
    # For more help, try 'type som_train_struct' or check out online documentation.
    # See also SOM_SET, SOM_TOPOL_STRUCT, SOM_MAKE.
    
    # DESCRIPTION
    #
    # This function is used to give sensible values for SOM training
    # parameters and returns a training struct. Often, the parameters
    # depend on the properties of the map and the training data. These are
    # given as optional arguments to the function. If a partially filled
    # train struct is given, its empty fields (field value is [] or '' or
    # NaN) are supplimented with default values.
    #
    # The training struct has a number of fields which depend on each other
    # and the optional arguments in complex ways. The most important argument 
    # is 'phase' which can be either 'init', 'train', 'rough' or 'finetune'.
    #
    #  'init'     Map initialization. 
    #  'train'    Map training in a onepass operation, as opposed to the
    #             rough-finetune combination.
    #  'rough'    Rough organization of the map: large neighborhood, big
    #             initial value for learning coefficient. Short training.
    #  'finetune' Finetuning the map after rough organization phase. Small
    #             neighborhood, learning coefficient is small already at 
    #             the beginning. Long training.
    #################################################
    
    ## check arguments

    # initial default structs
    sTrain = som_set('som_train')
    
    # initialize optional parameters
    dlen = np.nan
    msize = 0
    munits = np.nan
    sTprev =[]
    dim = np.nan
    phase = ['']
    breakpoint()
    i=0
    
    while i<= len(args)-1:
        argok = 1
        if isinstance(args[i], str):
            if args[i]=='dim':
                dim = kwargs[args[i]]
            elif args[i]=='dlen':
                dlen = kwargs[args[i]]
            elif args[i]== 'msize':
                msize = kwargs[args[i]]
            elif args[i] == 'munits':
                munits = kwargs[args[i]]
                msize = 0
            elif args[i]== 'phase':
                phase = kwargs[args[i]]
            elif args[i]== 'algorithm':
                sTrain['algorithm']= kwargs[args[i]]
            elif args[i]== 'mask':
                sTrain['mask']= kwargs[args[i]]
            # elif args[i] in ['previous', 'map']:
            #     if kwargs[args[i]]['type']== 'som_map':
                # continue in case it is needed!!!
            #         if len()
            elif args[i]== 'data':
                if isinstance(args[i], dict):
                    dlen, dim = kwargs[args[i]]['data'].shape
                else:
                    dlen, dim = kwargs[args[i]].shape
        elif isinstance(args[i], dict) and 'type' in args[i]:
            later=[]
            # continue later
        else:
            argok = 0
        
        if argok == 0:
            print(f'(som_train_struct) Ignoring invalid argument #{i + 1}')
        i= i+1
    # dim
    if not bool(sTprev) and np.isnan(dim):
        dim = len(sTprev['mask'])
    
    # mask
    if bool(sTrain['mask']) and not np.isnan(dim):
        sTrain['mask'] = np.ones((dim, 1))
    
    # msize, munits
    if msize==0 or not bool(msize):
        if np.isnan(munits):
            msize= (10,10)
        else:
            s= round(math.sqrt(munits))
            msize = (s, round(munits/s))
    munits = np.prod(msize)
    
    
    ## action
            
                    