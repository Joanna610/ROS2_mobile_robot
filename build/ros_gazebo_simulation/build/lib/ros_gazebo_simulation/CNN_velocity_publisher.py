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
import pandas as pd



def sighandler(signal, frame):
    # Po naciśnięciu Ctrl+C dane są zapisywane do CSV
    print("Otrzymano sygnał przerwania (Ctrl+C). Zapisuję dane...")
    combined_array = np.column_stack((lidar_subscriber.array_with_linear_velocity_real, lidar_subscriber.array_with_linear_velocity_prediction, 
                                        lidar_subscriber.array_with_angular_velocity_real, lidar_subscriber.array_with_angular_velocity_prediction))
    np.savetxt("CNN_output.csv", combined_array, delimiter=",", header="Predict_linear,Real_linear,Predict_angular,Real_angular", comments="", fmt="%.5f")
    print("Dane zapisane i ROS 2 zakończony.")


class LidarOdomSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_listener_callback,
            1)

        self.subscription_vel = self.create_subscription(
            Odometry,
            '/events/read_split',
            self.odom_listener_callback,
            10)
        # self.subscription  # prevent unused variable warning
        self.subscription_vel
        self.old_lidar_data = np.zeros(360)
        self.present_lidar_data = np.zeros(360)
        self.array_with_linear_velocity_prediction = np.zeros(780)
        self.array_with_angular_velocity_prediction = np.zeros(780)
        # self.array_with_linear_velocity_real = np.zeros(780)
        # self.array_with_angular_velocity_real = np.zeros(780)

        self.array_with_linear_velocity_real = np.zeros(10000)
        self.array_with_angular_velocity_real = np.zeros(10000)
        self.index_lidar = 0
        self.index_odom = 0
        
        self.linear_vel_model = models.load_model('/home/joanna/ros2_gz_sim/neural_networks/CNN_model_linear_velocity_Colab.keras')
        self.angular_vel_model = models.load_model('/home/joanna/ros2_gz_sim/neural_networks/CNN_model_angular_velocity_Colab.keras')
        self.predicted_velocity = Twist()


    def odom_listener_callback(self, msg):
        self.get_logger().info('Index listener: "%s"' % self.index)
        self.array_with_linear_velocity_real[self.index_odom] = msg.twist.twist.linear.x
        self.array_with_angular_velocity_real[self.index_odom] = msg.twist.twist.angular.z
        
        self.predicted_velocity.linear.x = self.array_with_linear_velocity_real[self.index_odom]
        self.predicted_velocity.angular.z = self.array_with_angular_velocity_real[self.index_odom]

        self.index_odom+=1
        

    def lidar_listener_callback(self, msg):
        # self.get_logger().info('Lidar listener: "%s"' % self.index)

        self.present_lidar_data = np.array(msg.ranges[:360])
        self.derivative_of_ranges = np.zeros(360)
        self.derivative_of_ranges[-1] = self.present_lidar_data[-1] - self.old_lidar_data[0]
        for i in range(359):
            self.derivative_of_ranges[i] = self.present_lidar_data[i] - self.old_lidar_data[i+1]

        self.old_lidar_data = self.present_lidar_data
        self.derivative_of_ranges = self.derivative_of_ranges.reshape(1, 360, 1)

        self.array_with_linear_velocity_prediction[self.index_lidar] = self.linear_vel_model.predict(self.derivative_of_ranges)
        self.array_with_angular_velocity_prediction[self.index_lidar]  = self.angular_vel_model.predict(self.derivative_of_ranges)

        self.get_logger().info('Prediction linear: "%s"' % self.array_with_linear_velocity_prediction[self.index_lidar])
        self.get_logger().info('Prediction angular: "%s"' % self.array_with_angular_velocity_prediction[self.index_lidar])
        
        self.predicted_velocity.linear.x = self.array_with_linear_velocity_prediction[self.index_lidar].item()
        self.predicted_velocity.angular.z = self.array_with_angular_velocity_prediction[self.index_lidar].item()
        self.index_lidar += 1
        self.publisher_.publish(self.predicted_velocity)

    

    
def main(args=None):
    signal.signal(signal.SIGINT, sighandler)
    rclpy.init(args=args)
    global lidar_odom_subscriber
    lidar_odom_subscriber = LidarOdomSubscriber()
   
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