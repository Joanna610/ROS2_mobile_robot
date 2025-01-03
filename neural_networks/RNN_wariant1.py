import os
import sys
import math
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tensorflow.keras import models, layers
from tensorflow.keras.layers import LSTM, GRU, Dense
# from ccnn_layers import CConv2D
from tensorflow.keras.callbacks import TensorBoard

import tensorflow as tf


def main():

    # raw data
    data = pd.read_csv("test_data.csv", dtype=float,  low_memory=False)
    data = np.array(data)
    T =  data.T

    velocity_linear = T[0]
    velocity_angular = T[1]
    ranges_T = T[2:]
    ranges = ranges_T.T
    m, n = ranges.shape # n - 360
    matrix_length = 4
    batch_size = 16
    epochs = 100

    amount_of_samples = 72
    half = 36
    array_with_samples = np.zeros((m, amount_of_samples))

    # array_with_samples = ranges.reshape(m, 5, 36)
    print("m: ", m)
    print("n: ", n)
    print("amount_of_samples: ", amount_of_samples)

    k = 0
    for i in range(0, m-1):
        for j in range(half):
            array_with_samples[k,j] = ranges[i, j*10]
        index = 0
        for j in range(half , amount_of_samples):
            array_with_samples[k,j] = ranges[i+1, index*10]
            index+=1
        k+=1

    print("Size: ",array_with_samples.shape)


    # =================== NEURAL NETWORK ===================
    model = models.Sequential()
    model.add(layers.Input(shape=(amount_of_samples, 1)))
    # model.add(GRU(50, return_sequences=True))
    # model.add(GRU(50))

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))

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

    #np.random.shuffle(linear_out)

    T =  linear_out.T
    velocity_linear = T[0]
    lidar_ranges = T[1:]

    # data ready to proccess
    lidar_ranges = lidar_ranges.T


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
    plt.grid(True)
    plt.show()

    model.save('RNN_GRU_model_linear_velocity_wariant_1.keras')

    # =================== ANGULAR VELOCITY ===================
    angular_out = np.vstack((velocity_angular, array_with_samples)).T

    #np.random.shuffle(angular_out)

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
    plt.plot(history.history['mae'], label='mae')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.ylim([0, max(history.history['loss'] + [max(history.history['mae'])])])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    model.save('RNN_GRU_model_angular_velocity_wariant_1.keras')

if __name__ == '__main__':
    main()