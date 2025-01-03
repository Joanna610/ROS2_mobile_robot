import csv
from matplotlib import pyplot as plt
import math

filename = '/home/joanna/ros2_gz_sim/RNN_GRU_var2_output_joint.csv'

# Parametry początkowe
x, y, theta, theta_real,theta_joint, x_real, y_real, joint_x, joint_y= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0
positions_x = [x]
positions_y = [y]
positions_x_real = [x_real]
positions_y_real = [y_real]
positions_x_joint = [joint_x]
positions_y_joint = [joint_y]
delta_t = 0.1  # odstęp czasowy między próbkami to 0.1 sekundy
radius = 0.035
distance_between = 0.2

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    
    for row in reader:
        linear_velocity = float(row[1])  # Prędkość liniowa
        angular_velocity = float(row[3])  # Prędkość kątowa

        linear_velocity_real = float(row[0])  # Prędkość liniowa
        angular_velocity_real = float(row[2])  # Prędkość kątowa

        joint_L = float(row[4])*radius  # Prędkość pierwszego koła
        joint_R = float(row[5])*radius  # Prędkość drugiego koła

        v = (joint_L + joint_R) / 2
        omega = (joint_R - joint_L) / distance_between
        
        # Aktualizacja orientacji
        theta += angular_velocity * delta_t
        theta_real += angular_velocity_real * delta_t
        theta_joint += omega * delta_t
        

        # Aktualizacja pozycji
        x += linear_velocity * math.cos(theta) * delta_t
        y += linear_velocity * math.sin(theta) * delta_t

        # Aktualizacja pozycji
        x_real += linear_velocity_real * math.cos(theta_real) * delta_t
        y_real += linear_velocity_real * math.sin(theta_real) * delta_t

        #Aktualizacja pozycji
        joint_x += v * math.cos(theta_joint) * delta_t
        joint_y += v * math.sin(theta_joint) * delta_t
        

        # Zapisz nowe pozycje
        positions_x.append(x)
        positions_y.append(y)

        # Zapisz nowe pozycje
        positions_x_real.append(x_real)
        positions_y_real.append(y_real)

        # Zapisz nowe pozycje
        positions_x_joint.append(joint_x)
        positions_y_joint.append(joint_y)

# Rysowanie trajektorii robota
plt.figure(figsize=(10, 6))
plt.plot(positions_x_real, positions_y_real, label='rzeczywista trajektoria')
plt.plot(positions_x, positions_y, label='przewidziana trajektoria', color='purple')
plt.plot(positions_x_joint, positions_y_joint, label='odometryczna trajektoria',  linestyle='dashed', color='gray')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
