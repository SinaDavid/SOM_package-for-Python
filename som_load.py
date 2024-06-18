import numpy as np
from som_side_functions import *

def som_data_struct(D, name=None, *varargin):
    """
    SOM_DATA_STRUCT Create a data struct.

    sData = som_data_struct(D, [argID, value, ...])

     sData  = som_data_struct(D); 
     sData  = som_data_struct(D,'name','my_data','labels',labs);

     Input and output arguments ([]'s are optional): 
      D        (matrix) data matrix, size dlen x dim
      [argID,  (string) See below. These are given as argID, value pairs.
       value]  (varies) 

      sData    (struct) created data struct

     Here are the argument IDs and corresponding values: 
      'labels'     (string array / cellstr) labels for each data vector,
                    length=dlen
      'name'       (string) data name
      'comp_names' (string array / cellstr) component names, size dim x 1
      'comp_norm'  (cell array) normalization operations for each
                    component, size dim x 1. Each cell is either empty, 
                    or a cell array of normalization structs.

    For more help, try 'type som_data_struct' or check out online documentation.
    See also SOM_SET, SOM_INFO, SOM_MAP_STRUCT.
    """
    
    # Get size of input data
    dlen, dim = D.shape
    # Initialize name
    if name is None:
        name = 'unnamed'

    # Initialize labels
    labels = [''] * dlen

    # Create comp_names list
    comp_names = [f'Variable{i}' for i in range(1, dim+1)]

    # Create an empty list of length dim for comp_norm
    comp_norm = [''] * dim

    # Initialize label_names
    label_names = []

    # Iterating over varargin
    i = 0
    while i < len(varargin):
        argok = True
        if isinstance(varargin[i], str):
            if varargin[i] == 'comp_names':
                i += 1
                comp_names = varargin[i]
            elif varargin[i] == 'labels':
                i += 1
                labels = varargin[i]
            elif varargin[i] == 'name':
                i += 1
                name = varargin[i]
            elif varargin[i] == 'comp_norm':
                i += 1
                comp_norm = varargin[i]
            elif varargin[i] == 'label_names':
                i += 1
                label_names = varargin[i]
            else:
                argok = False
        else:
            argok = False

        if not argok:
            print(f'(som_data_struct) Ignoring invalid argument #{i+1}')

        i += 1

    # Create struct (represented as a dictionary)
    sData = som_set('som_data',*['data', 'labels', 'name', 'comp_names', 'comp_norm', 'label_names'],**{
        'data': D,
        'labels': labels,
        'name': name,
        'comp_names': comp_names,
        'comp_norm': comp_norm,
        'label_names': label_names
    })
    
        #             sData = som_set('som_data','data','labels','name','comp_names','comp_norm','label_names','data'=D, 'labels'=labels,'name'= name,
        # 'comp_names'= comp_names,'comp_norm'= comp_norm,'label_names'= label_names)

    return sData
