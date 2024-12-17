
import os
import sys
import math
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import models, layers
# from ccnn_layers import CConv2D
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf

def main():
    # raw data
    data = pd.read_csv("test_data.csv", dtype=float,  low_memory=False)
    data = np.array(data)
    # m, n = data.shape
    T =  data.T

    velocity_linear = T[0]
    velocity_angular = T[1]
    ranges_T = T[2:]    
    ranges = ranges_T.T
    new_m, new_n = ranges.shape
    derivative_of_ranges_for_angular = np.zeros((new_m-1, new_n))
    derivative_of_ranges_for_linear = np.zeros((new_m-1, new_n))
    batch_size = 16
    epochs = 50

    # remove first value of velocity
    velocity_linear = np.delete(velocity_linear, 0)
    velocity_angular = np.delete(velocity_angular, 0)

    # calculate derivative for angular velocity - changes
    for i in range(1, ranges_T[0].size):
        derivative_of_ranges_for_angular[i-1][-1] = ranges[i][-1] - ranges[i-1][0]
        for j in range(new_n-1):
            derivative_of_ranges_for_angular[i-1][j] = ranges[i][j] - ranges[i-1][j+1]

    # calculate derivative for linear velocity - changes
    for i in range(1, ranges_T[0].size):
        for j in range(new_n):
            derivative_of_ranges_for_linear[i-1][j] = ranges[i][j] - ranges[i-1][j]

     # building model
    model = models.Sequential()
    model.add(layers.Input(shape=(360,1)))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
        )

    # =================== LINEAR VELOCITY ===================
    # merge to shuffle
    derivative_of_ranges_for_linear = derivative_of_ranges_for_linear.T
    linear_out = np.vstack((velocity_linear, derivative_of_ranges_for_linear)).T

    np.random.shuffle(linear_out) 
    
    # df = pd.DataFrame(linear_out)

    # # Zapisujemy jako CSV
    # df.to_csv('Walidation2_linear.csv', index=False)

    T =  linear_out.T
    velocity_linear = T[0]
    lidar_ranges = T[1:]    

    # data ready to proccess
    lidar_ranges = lidar_ranges.T


    # DATA AUGMENTATION
    # percent_value = 0.05
    # gauss_noise = tf.random.normal(shape=tf.shape(lidar_ranges), mean=0.0, stddev=percent_value*max_lidar_value, dtype=tf.float64)
    # lidar_ranges_with_noise = tf.add(lidar_ranges, gauss_noise)

    history = model.fit(lidar_ranges, velocity_linear,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            )


    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['mae'], label='mae')  # Jeśli masz MAE jako metrykę
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.ylim([0, max(history.history['loss'] + [max(history.history['mae'])])])  # Ustaw limity Y
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    model.save('CNN_model_linear_velocity.keras')


    # =================== ANGULAR VELOCITY ===================
    # merge to shuffle
    derivative_of_ranges_for_angular = derivative_of_ranges_for_angular.T
    angular_out = np.vstack((velocity_angular, derivative_of_ranges_for_angular)).T

    np.random.shuffle(angular_out) 
    
    # df = pd.DataFrame(angular_out)

    # # Zapisujemy jako CSV
    # df.to_csv('Walidation2_angular.csv', index=False)

    T =  angular_out.T
    velocity_angular = T[0]
    lidar_ranges = T[1:]    

    # data ready to proccess
    lidar_ranges = lidar_ranges.T
   

    history = model.fit(lidar_ranges, velocity_angular,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['mae'], label='mae')  # Jeśli masz MAE jako metrykę
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.ylim([0, max(history.history['loss'] + [max(history.history['mae'])])])  # Ustaw limity Y
    plt.legend(loc='upper right')
    plt.show()

    model.save('CNN_model_angular_velocity.keras')

if __name__ == '__main__':
    main()