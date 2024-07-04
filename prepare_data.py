# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:29:54 2024

@author: sdd380
"""

import os
os.chdir('C:/Users/sdd380/surfdrive - David, S. (Sina)@surfdrive.surf.nl/Projects/SOM_Workshop/SOM-Workshop/')
import subprocess
import glob
import numpy as np
import pandas as pd
from scipy.signal import decimate
from normalize import normalize
import matplotlib.pyplot as plt

def replace_nan_with_average(df):
    for column in df.columns:
        col_data = df[column].values
        i = 0
        while i < len(col_data):
            if np.isnan(col_data[i]):
                # Find the first non-NaN value before NaN
                j = i - 1
                while j >= 0 and np.isnan(col_data[j]):
                    j -= 1
                prev_val = col_data[j] if j >= 0 else np.nan
                
                # Find the first non-NaN value after NaN
                k = i + 1
                while k < len(col_data) and np.isnan(col_data[k]):
                    k += 1
                next_val = col_data[k] if k < len(col_data) else np.nan
                
                # Replace NaN with the average of neighboring values
                if not np.isnan(prev_val) and not np.isnan(next_val):
                    col_data[i] = (prev_val + next_val) / 2
                elif not np.isnan(prev_val):
                    col_data[i] = prev_val
                elif not np.isnan(next_val):
                    col_data[i] = next_val
                else:
                    col_data[i] = 0  # Default to 0 if both neighbors are NaN
                # Move to the next position after replacement
                i = k
            else:
                i += 1
        df[column] = col_data
    return df

# Apply the function to the DataFrame

# path = 'C:\\Users\\mario\\OneDrive - UWA\\ISBS2024\\'
path = 'C:\\Users\\sdd380\\Downloads\\4543435\\'
marker_files = sorted(glob.glob(path + 'markers\\*.csv'))
event_files = sorted(glob.glob(path + 'events\\*.csv'))

marker_files = [i for i in marker_files if 'run' in i or "walk" in i]
event_files = [i for i in event_files if 'run' in i or "walk" in i]
all_data_norm = []


for marker_file, event_file in zip(marker_files, event_files):
    assert(marker_file.split('\\')[-1].split('.')[0] == event_file.split('\\')[-1].split('.')[0])
    name = marker_file.split('\\')[-1].split('.')[0]
    
    # markers
    data = pd.read_csv(marker_file)
    data.columns = data.columns.str.lower()
    selected_columns_x = data[['r.asis_x', 'r.psis_x', 'l.asis_x', 'l.psis_x']]
    selected_columns_y = data[['r.asis_y', 'r.psis_y', 'l.asis_y', 'l.psis_y']]
    selected_columns_z = data[['r.asis_z', 'r.psis_z', 'l.asis_z', 'l.psis_z']]


    # Calculate the mean of the selected columns
    mean_selected_columns_x = selected_columns_x.mean(axis=1)
    mean_selected_columns_y = selected_columns_y.mean(axis=1)
    mean_selected_columns_z = selected_columns_z.mean(axis=1)
    
    cols_to_modify_x = data.columns[::3]  # This will give ['A', 'D']
    cols_to_modify_y = data.columns[1::3]
    cols_to_modify_z = data.columns[2::3]
    
    # Subtract the row-wise mean from these columns
    for col in cols_to_modify_x:
        data[col] = data[col] - mean_selected_columns_x
        
    for col in cols_to_modify_y:
        data[col] = data[col] - mean_selected_columns_y
    
    
    for col in cols_to_modify_z:
        data[col] = data[col] - mean_selected_columns_z
    # events
    event = pd.read_csv(event_file)
    data_segments = []
    for i in range(0,10):
        ## segment and normalise data, include 10 gait cycles per person
        data_temp = data.iloc[event['Events'].iloc[i]:event['Events'].iloc[i + 1], :]
        data_norm_temp = normalize(data_temp,1)
        data_segments.append(data_norm_temp)
        
    data_norm = np.vstack(data_segments)
    all_data_norm.append(data_norm)

# Vertically concatenate all data_norm arrays from all iterations
final_data_norm = np.vstack(all_data_norm)

# Convert final_data_norm to a DataFrame
final_df = pd.DataFrame(final_data_norm, columns=data.columns)

# Save the DataFrame to a CSV file
# final_df.to_csv('running.csv', index=False)


path2 = 'C:\\Users\\sdd380\\surfdrive - David, S. (Sina)@surfdrive.surf.nl\\Data_stroke\\HealthyData\\Dyn\\'
marker_files = sorted(glob.glob(path2 + 'marker\\*.csv'))
event_files = sorted(glob.glob(path2 + 'events\\*.csv'))

marker_files = [i for i in marker_files if 'run' in i or "walk" in i]
event_files = [i for i in event_files if 'run' in i or "walk" in i]
all_data_norm = []


for marker_file, event_file in zip(marker_files, event_files):
    assert(marker_file.split('\\')[-1].split('.')[0] == event_file.split('\\')[-1].split('.')[0])
    name = marker_file.split('\\')[-1].split('.')[0]
    
    # markers
    data1 = pd.read_csv(marker_file)
    data1.columns = data1.columns.str.lower()
    data1=data1[data.columns]
    
    selected_columns_x1 = data1[['r.asis_x', 'r.psis_x', 'l.asis_x', 'l.psis_x']]
    selected_columns_y1 = data1[['r.asis_y', 'r.psis_y', 'l.asis_y', 'l.psis_y']]
    selected_columns_z1 = data1[['r.asis_z', 'r.psis_z', 'l.asis_z', 'l.psis_z']]


    # Calculate the mean of the selected columns
    mean_selected_columns_x1 = selected_columns_x1.mean(axis=1)
    mean_selected_columns_y1 = selected_columns_y1.mean(axis=1)
    mean_selected_columns_z1 = selected_columns_z1.mean(axis=1)

    cols_to_modify_x = data1.columns[::3]  # This will give ['A', 'D']
    cols_to_modify_y = data1.columns[1::3]
    cols_to_modify_z = data1.columns[2::3]
    
    # Subtract the row-wise mean from these columns
    for col in cols_to_modify_x:
        data1[col] = data1[col] - mean_selected_columns_x1
        
    for col in cols_to_modify_y:
        data1[col] = data1[col] - mean_selected_columns_y1
    
    
    for col in cols_to_modify_z:
        data1[col] = data1[col] - mean_selected_columns_z1
    # events
    event1 = pd.read_csv(event_file)
    data_segments = []
    for i in range(1,11):
        ## segment and normalise data, include 10 gait cycles per person
        
        data_temp = data1.iloc[event['Events'].iloc[i]:event['Events'].iloc[i + 1],:]
        data_temp = replace_nan_with_average(data_temp)
        data_norm_temp = normalize(data_temp,1)
        data_segments.append(data_norm_temp)
        
    data_norm = np.vstack(data_segments)
    all_data_norm.append(data_norm)

# Vertically concatenate all data_norm arrays from all iterations
final_data_norm1 = np.vstack(all_data_norm)

# Convert final_data_norm to a DataFrame
final_df1 = pd.DataFrame(final_data_norm1, columns=data.columns)

# Save the DataFrame to a CSV file
final_df1.to_csv('walking.csv', index=False)

    # plt.plot(data_norm[:,0])
    # plt.title('Plot of the First Column')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()
    
    # plt.plot(data_temp.iloc[:,0])
    # plt.title('Plot of the First Column')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()

