import os
import sys
import math
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers
import tensorflow as tf


def main():

    if len(sys.argv) < 3:
        print("Podano niewystarczajaca ilosc argumentow.")
        return -1

    data = pd.read_csv(sys.argv[1], dtype=float,  low_memory=False)
    data = np.array(data)
    T =  data.T

    velocity_linear = T[0]
    # velocity_angular = T[1]
    ranges_T = T[1:]

    ranges = ranges_T.T

    # ----------------------------- LINEAR VELOCITY -----------------------------
    new_model = models.load_model('CNN_model_linear_velocity.keras')
    error = np.zeros((velocity_linear.shape))

    # print(single_measurement)
    for i in range(ranges.shape[0]):
        single_measurement_from_lidar = ranges[i].reshape(1, -1)
        vel_prediction = new_model.predict(single_measurement_from_lidar)
        error[i] = velocity_linear[i] - vel_prediction

    test_loss, test_acc = new_model.evaluate(ranges,  velocity_linear, verbose=2)

    print("loss: ", test_loss)
    print("accuracy: ", test_acc)

    # Tworzenie histogramu z błędami
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=200)  
    plt.xlabel('Wartość błędu')
    plt.ylabel('Liczba wystąpień')
    plt.grid(True)
    plt.show()


    # ----------------------------- ANGULAR VELOCITY -----------------------------
    data = pd.read_csv(sys.argv[2], dtype=float,  low_memory=False)
    data = np.array(data)
    T =  data.T

    velocity_angular = T[0]
    ranges_T = T[1:]

    ranges = ranges_T.T

    new_model = models.load_model('/home/joanna/Downloads/CNN_model_angular_velocity_new_100.keras')
    error = np.zeros((velocity_linear.shape))

    # print(single_measurement)
    for i in range(ranges.shape[0]):
        single_measurement_from_lidar = ranges[i].reshape(1, -1)
        vel_prediction = new_model.predict(single_measurement_from_lidar)
        error[i] = velocity_angular[i] - vel_prediction

    test_loss, test_acc = new_model.evaluate(ranges,  velocity_angular, verbose=2)

    print("loss: ", test_loss)
    print("accuracy: ", test_acc)

    # Tworzenie histogramu z błędami
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=200)  
    plt.xlabel('Wartość błędu')
    plt.ylabel('Liczba wystąpień')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()