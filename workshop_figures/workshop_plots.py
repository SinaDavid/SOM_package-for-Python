# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:00:20 2024

@author: sdd380
"""

## make some plots for workshop
import numpy as np
import matplotlib.pyplot as plt


markers_train = np.reshape(data[:,0],(len(data)//101,101))
markers = markers_train.T


markers_test = np.reshape(test_data[:,0],(len(test_data)//101,101))
markers_t = markers_test.T

x = np.arange(1, 102)
# Create the plot
plt.figure()
p1, = plt.plot(x, markers[:,1], 'b', linewidth=4, label='train Mean')
p2, = plt.plot(x, markers_t[:,1], 'r', linewidth=4, label='test Mean')

plt.show()


train = np.reshape(data[:, 0], (101, -1), order='F')
MEAN_marker_train = np.mean(train,axis=1)
STD_marker_train = np.std(train, axis=1)


test = np.reshape(test_data[:, 0], (101, -1), order='F')
MEAN_marker_test = np.mean(test,axis=1)
STD_marker_test = np.std(test, axis=1)

x = np.arange(1, 102)
y = STD_marker_train
y1 = STD_marker_test
# Create the plot
plt.figure()
p1, = plt.plot(x, MEAN_marker_train, 'k', linewidth=4, label='train Mean')
plt.fill_between(x, MEAN_marker_train - y , MEAN_marker_train + y, color=[0.6, 0.7, 0.8], alpha=0.2)
p2, = plt.plot(x, MEAN_marker_test, 'r', label='test_MEAN')
plt.fill_between(x, MEAN_marker_test - y1 , MEAN_marker_test + y1, color=[0.8, 0.1, 0.1], alpha=0.2)


plt.xlim([1, 101])
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Quantization Error')
plt.legend()

plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot data in each subplot
for i in range(6):
    train = np.reshape(data[:, i], (101, -1), order='F')
    MEAN_marker_train = np.mean(train,axis=1)
    STD_marker_train = np.std(train, axis=1)


    test = np.reshape(test_data[:, i], (101, -1), order='F')
    MEAN_marker_test = np.mean(test,axis=1)
    STD_marker_test = np.std(test, axis=1)
    
    x = np.arange(1, 102)
    y = STD_marker_train
    y1 = STD_marker_test
    
    axes[i].plot(x, MEAN_marker_train, 'k', linewidth=4, label='train Mean')  # Example: Different sine wave for each subplot
    axes[i].fill_between(x, MEAN_marker_train - y , MEAN_marker_train + y, color=[0.6, 0.7, 0.8], alpha=0.2)
    axes[i].plot(x, MEAN_marker_test, 'r', label='test_MEAN')  # Example: Different sine wave for each subplot
    axes[i].fill_between(x, MEAN_marker_test - y1 , MEAN_marker_test + y1, color=[0.8, 0.1, 0.1], alpha=0.2)

    axes[i].set_title(headers[i])
    axes[i].set_xlabel('gait cycle (%)')
    axes[i].set_ylabel('Displacement (mm)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


#################################
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot data in each subplot
for i in range(9):
    train = np.reshape(data[:, i], (101, -1), order='F')
    MEAN_marker_train = np.mean(train,axis=1)
    STD_marker_train = np.std(train, axis=1)


    test = np.reshape(data[:, i+9], (101, -1), order='F')
    MEAN_marker_test = np.mean(test,axis=1)
    STD_marker_test = np.std(test, axis=1)
    
    x = np.arange(1, 102)
    y = STD_marker_train
    y1 = STD_marker_test
    
    axes[i].plot(x, MEAN_marker_train, 'k', linewidth=4, label='train Mean')  # Example: Different sine wave for each subplot
    axes[i].fill_between(x, MEAN_marker_train - y , MEAN_marker_train + y, color=[0.6, 0.7, 0.8], alpha=0.2)
    axes[i].plot(x, MEAN_marker_test, 'r', label='test_MEAN')  # Example: Different sine wave for each subplot
    axes[i].fill_between(x, MEAN_marker_test - y1 , MEAN_marker_test + y1, color=[0.8, 0.1, 0.1], alpha=0.2)

    axes[i].set_title(headers[i])
    axes[i].set_xlabel('gait cycle (%)')
    axes[i].set_ylabel('Displacement (mm)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()



# Calculate the mean and standard deviation
mean_train = np.mean(MEAN_error_train, axis=0)
std_train = np.std(MEAN_error_train, axis=0)
mean_test = np.mean(MEAN_error_test, axis=0)
std_test = np.std(MEAN_error_test, axis=0)
fig, ax = plt.subplots(figsize=(6, 6))

# Plot mean and standard deviation as error bars
ax.errorbar(1, mean_train, yerr=std_train, fmt='bo', label='Train Mean X')
ax.errorbar(1, mean_test, yerr=std_test, fmt='ro', label='Test Mean X')

# Customize the plot
ax.set_xticks([1, 1])
ax.set_xticklabels(['Train', 'Test'])
ax.set_ylabel('Quantization Error')
ax.set_title('Mean and Standard Deviation of Train and Test Errors')
ax.legend()

# Show the plot
plt.show()
