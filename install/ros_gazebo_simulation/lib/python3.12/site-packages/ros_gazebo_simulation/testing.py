# import pandas as pd
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
from tensorflow.keras import models
import signal
import rclpy
from rclpy.node import Node
import csv
import os
import sys




# def sighandler(signal, frame):
#     # Po naciśnięciu Ctrl+C dane są zapisywane do CSV
#     print("Otrzymano sygnał przerwania (Ctrl+C). Zapisuję dane...")
#     combined_array = np.column_stack((lidar_subscriber.array_with_linear_velocity_real, lidar_subscriber.array_with_linear_velocity_prediction, 
#                                         lidar_subscriber.array_with_angular_velocity_real, lidar_subscriber.array_with_angular_velocity_prediction))
#     np.savetxt("CNN_output.csv", combined_array, delimiter=",", header="Predict_linear,Real_linear,Predict_angular,Real_angular", comments="", fmt="%.5f")
#     print("Dane zapisane i ROS 2 zakończony.")

def sighandler(signal, frame):
    # Po naciśnięciu Ctrl+C dane są zapisywane do CSV
    print("Otrzymano sygnał przerwania (Ctrl+C). Zapisuję dane...")
    combined_array = np.column_stack((lidar_odom_subscriber.array_with_linear_velocity_real, 
                                        lidar_odom_subscriber.array_with_angular_velocity_real))
    np.savetxt("output.csv", combined_array, delimiter=",", header="Predict_linear, angular")
    print("Dane zapisane i ROS 2 zakończony.")

class Test(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        filename = '/home/joanna/ros2_gz_sim//bag_folder20241126_144849/rosbag_data_20241126_144849.csv'
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            
            self.col1 = []
            self.col2 = []

            for row in reader:
                self.col1.append(row[0])
                self.col2.append(row[1])

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        self.array_with_linear_velocity_real = np.zeros(780)
        self.array_with_angular_velocity_real = np.zeros(780)
        self.predicted_velocity = Twist()

    def timer_callback(self):
        self.predicted_velocity.linear.x = float(self.col1[sevelocity_linearlf.i])
        self.predicted_velocity.angular.z = float(self.col2[self.i])

        self.array_with_linear_velocity_real[self.i] = self.predicted_velocity.linear.x
        self.array_with_angular_velocity_real[self.i] = self.predicted_velocity.angular.z

        self.publisher_.publish(self.predicted_velocity)
        self.get_logger().info('Publishing vel1: "%s"' % self.predicted_velocity.linear.x)
        self.get_logger().info('Publishing vel2: "%s"' % self.predicted_velocity.angular.z)
        self.i += 1

    

    
def main(args=None):
    signal.signal(signal.SIGINT, sighandler)
    rclpy.init(args=args)
    global lidar_odom_subscriber
    lidar_odom_subscriber = Test()
   
    rclpy.spin(lidar_odom_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # combined_array = np.column_stack((lidar_subscriber.array_with_linear_velocity_prediction, lidar_subscriber.array_with_angular_velocity_prediction))
    # np.savetxt("output.csv", combined_array, delimiter=",", header="Column1,Column2", comments="", fmt="%.5f")
    lidar_odom_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()