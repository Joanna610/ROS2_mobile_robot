import csv
from matplotlib import pyplot as plt

filename = '/home/joanna/ros2_gz_sim/CNN_output.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    
    real_linear = []
    predicted_linear = []
    real_angular = []
    predicted_angular = []
    real_linear_position = []
    predicted_linear_position = []
    real_angular_position = []
    predicted_angular_position = []

    for row in reader:
        real_linear.append(float(row[0]))
        predicted_linear.append(float(row[1]))
        real_angular.append(float(row[2]))
        predicted_angular.append(float(row[3]))

        real_linear_position.append(float(row[6]))
        predicted_linear_position.append(float(row[7]))
        real_angular_position.append(float(row[8]))
        predicted_angular_position.append(float(row[9]))
        
plt.figure(figsize=(10, 6))
plt.plot(real_linear, label='rzeczywista')
plt.plot(predicted_linear, label='przewidziana', color='purple')
plt.xlabel('Czas')
plt.ylabel('Prędkość')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(real_angular, label='rzeczywista')
plt.plot(predicted_angular, label='przewidziana', color='purple')
plt.xlabel('Czas')
plt.ylabel('Prędkość')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
        
plt.figure(figsize=(10, 6))
plt.plot(real_linear_position, label='rzeczywiste')
plt.plot(predicted_linear_position, label='przewidziane', color='purple')
plt.xlabel('Czas')
plt.ylabel('Położenie')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(real_angular_position, label='rzeczywiste')
plt.plot(predicted_angular_position, label='przewidziane', color='purple')
plt.xlabel('Czas')
plt.ylabel('Położenie')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
