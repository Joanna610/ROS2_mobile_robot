
import os
import sys
import math
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tensorflow.keras import models, layers
from tensorflow.keras.layers import LSTM, Dense, GRU
# from ccnn_layers import CConv2D
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():

    if len(sys.argv) < 2:
        print("Nie podano nazwy pliku.")
        return -1

    # raw data
    data = pd.read_csv(sys.argv[1], dtype=float,  low_memory=False)
    data = np.array(data)
    T =  data.T

    velocity_linear = T[0]
    velocity_angular = T[1]
    ranges_T = T[2:]    
    ranges = ranges_T.T
    m, n = ranges.shape # n - 360
    matrix_length = 4
    batch_size = 16
    epochs = 80

    amount_of_samples = 36
    array_with_samples = np.zeros((m, amount_of_samples))
    # array_with_samples = ranges.reshape(m, 5, 36)    
    
    for i in range(m):
        for j in range(amount_of_samples):
            array_with_samples[i,j] = ranges[i, j*10]


    model = models.Sequential()
    model.add(layers.Input(shape=(amount_of_samples, 1)))
    model.add(GRU(50, return_sequences=True, dropout=0.2))
    model.add(GRU(50, dropout=0.2))

    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

     # =================== LINEAR VELOCITY ===================
    array_with_samples = array_with_samples.T
    linear_out = np.vstack((velocity_linear, array_with_samples)).T

    # np.random.shuffle(linear_out) 

    T =  linear_out.T
    velocity_linear = T[0]
    lidar_ranges = T[1:]    

    # data ready to proccess
    lidar_ranges = lidar_ranges.T

     # DATA AUGMENTATION
    # max_lidar_value=40
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
    plt.plot(history.history['mae'], label='mae') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.ylim([0, max(history.history['loss'] + [max(history.history['mae'])])]) 
    plt.legend(loc='upper right')
    plt.title('Wykres strat i błędów dla prędkości liniowych')
    plt.grid(True)
    plt.show()
    
    model.save('RNN_model_linear_velocity.keras')

    # =================== ANGULAR VELOCITY ===================
    angular_out = np.vstack((velocity_angular, array_with_samples)).T

    # np.random.shuffle(angular_out) 

    T =  angular_out.T
    velocity_angular = T[0]
    lidar_ranges = T[1:]    

    # data ready to proccess
    lidar_ranges = lidar_ranges.T

     # DATA AUGMENTATION
    # max_lidar_value=40
    # percent_value = 0.05
    # gauss_noise = tf.random.normal(shape=tf.shape(lidar_ranges), mean=0.0, stddev=percent_value*max_lidar_value, dtype=tf.float64)
    # lidar_ranges_with_noise = tf.add(lidar_ranges, gauss_noise)

    history = model.fit(lidar_ranges, velocity_angular,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['mae'], label='mae') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.ylim([0, max(history.history['loss'] + [max(history.history['mae'])])]) 
    plt.legend(loc='upper right')
    plt.title('Wykres strat i błędów dla prędkości kątowych')
    plt.grid(True)
    plt.show()

    model.save('RNN_model_angular_velocity.keras')

if __name__ == '__main__':
    main()