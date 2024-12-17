import os
import sys
import math
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers
import tensorflow as tf

data = pd.read_csv(sys.argv[1], dtype=float,  low_memory=False)
data = np.array(data)

T =  data.T

velocity_linear = T[0]
velocity_angular = T[1]
ranges_T = T[2:]

ranges = ranges_T.T
m, n = ranges.shape


error = np.zeros((velocity_linear.shape))
amount_of_samples = 36

array_with_samples = np.zeros((m, amount_of_samples))

# prepare data
for i in range(m):
    for j in range(amount_of_samples):
        array_with_samples[i,j] = ranges[i, j*10]


# # ----------------------- LINEAR VELOCITY -----------------------
new_model = models.load_model('RNN_GRU_model_linear_velocity_Colab.keras')

# print(single_measurement)
for i in range(ranges.shape[0]):
    single_measurement_from_lidar = array_with_samples[i].reshape(1, -1)
    vel_prediction = new_model.predict(single_measurement_from_lidar)
    error[i] = velocity_linear[i] - vel_prediction

print("array_with_samples: ", array_with_samples.shape)
print("velocity_linear: ", velocity_linear.shape)

test_loss, test_acc = new_model.evaluate(array_with_samples,  velocity_linear, verbose=2)

print("loss: ", test_loss)
print("accuracy: ", test_acc)

# Tworzenie histogramu z błędami
plt.figure(figsize=(10, 6))
plt.hist(error, bins=200)  
plt.xlabel('Wartość błędu')
plt.ylabel('Liczba wystąpień')
plt.grid(True)
plt.show()


# =====================================================================================================================
# =====================================================================================================================

# ----------------------- ANGULAR VELOCITY -----------------------
new_model = models.load_model('RNN_GRU_model_angular_velocity_Colab.keras')

for i in range(ranges.shape[0]):
    single_measurement_from_lidar = array_with_samples[i].reshape(1, -1)
    vel_prediction = new_model.predict(single_measurement_from_lidar)
    error[i] = velocity_angular[i] - vel_prediction

test_loss, test_acc = new_model.evaluate(array_with_samples, velocity_angular, verbose=2)

print("loss: ", test_loss)
print("accuracy: ", test_acc)

# Tworzenie histogramu z błędami
plt.figure(figsize=(10, 6))
plt.hist(error, bins=200)  
plt.xlabel('Wartość błędu')
plt.ylabel('Liczba wystąpień')
plt.grid(True)
plt.show()


