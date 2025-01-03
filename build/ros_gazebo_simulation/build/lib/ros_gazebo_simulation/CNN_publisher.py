import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
from tensorflow.keras import models
import signal
import rclpy
from rclpy.node import Node
import csv

# import pandas as pd



def sighandler(signal, frame):
    # Po naciśnięciu Ctrl+C dane są zapisywane do CSV
    print("Otrzymano sygnał przerwania (Ctrl+C). Zapisuję dane...")
    combined_array = np.column_stack((lidar_odom_subscriber.array_with_linear_velocity_real, lidar_odom_subscriber.array_with_linear_velocity_prediction, 
                                        lidar_odom_subscriber.array_with_angular_velocity_real, lidar_odom_subscriber.array_with_angular_velocity_prediction,
                                        lidar_odom_subscriber.joint_velocities_x, lidar_odom_subscriber.joint_velocities_z))
    np.savetxt("CNN_output_with_joints.csv", combined_array, delimiter=",", header="Real_linear, Prediction_linear, Real_angular, Prediction_angular,  joint_state_velocity_x, joint_state_velocity_y")
    print("Dane zapisane i ROS 2 zakończony.")


class LidarOdomSubscriber(Node):

    def __init__(self):
        super().__init__('lidar_subscriber')

        # velocity publisher and timer
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.joint_subscriber = self.create_subscription(
            JointState,
            '/joint_states',  # Topic to listen
            self.joint_callback,
            10
        )
        self.joint_subscriber
        self.joint_names = []
        self.joint_position = []
        self.joint_velocities_x = np.zeros(500)
        self.joint_velocities_z = np.zeros(500)


        # open file
        filename = '/home/joanna/ros2_gz_sim/bag_folder20241126_144849/rosbag_data_20241126_144849.csv'
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            
            self.velocity_linear = []
            self.velocity_angular = []
            self.lidar_data = []

            for row in reader:
                self.velocity_linear.append(row[0])
                self.velocity_angular.append(row[1])
                self.lidar_data.append([float(x) for x in row[2:362]])

        # keep lidar data
        self.old_lidar_data = [0] * 360
        self.present_lidar_data = [0] * 360

        # keep prediction velocity
        self.array_with_linear_velocity_prediction = np.zeros(500)
        self.array_with_angular_velocity_prediction = np.zeros(500)

        # keep real velocity
        self.array_with_linear_velocity_real = np.zeros(500)
        self.array_with_angular_velocity_real = np.zeros(500)
        
        self.linear_vel_model = models.load_model('/home/joanna/ros2_gz_sim/neural_networks/CNN_model_linear_velocity.keras')
        self.angular_vel_model = models.load_model('/home/joanna/ros2_gz_sim/neural_networks/CNN_model_angular_velocity.keras')
        self.predicted_velocity = Twist()
        

    def timer_callback(self):
        self.derivative_of_ranges = np.zeros(360)
        self.present_lidar_data = self.lidar_data[self.i]

        # calculate the derivative
        self.derivative_of_ranges[-1] = self.present_lidar_data[-1] - self.old_lidar_data[0]
        for i in range(359):
            self.derivative_of_ranges[i] = self.present_lidar_data[i] - self.old_lidar_data[i+1]
        # exchange new value with old
        self.old_lidar_data = self.present_lidar_data
        self.derivative_of_ranges = self.derivative_of_ranges.reshape(1, 360, 1)


        # make prediction
        self.array_with_linear_velocity_prediction[self.i] = self.linear_vel_model.predict(self.derivative_of_ranges)
        self.array_with_angular_velocity_prediction[self.i]  = self.angular_vel_model.predict(self.derivative_of_ranges)

        self.get_logger().info('Publishing vel1: "%s"' % self.array_with_linear_velocity_prediction[self.i])
        self.get_logger().info('Publishing vel2: "%s"' % self.array_with_angular_velocity_prediction[self.i]) 

        # publishing prediction
        self.predicted_velocity.linear.x = self.array_with_linear_velocity_prediction[self.i].item()
        self.predicted_velocity.angular.z = self.array_with_angular_velocity_prediction[self.i].item()
        self.publisher_.publish(self.predicted_velocity)

        self.array_with_linear_velocity_real[self.i] = float(self.velocity_linear[self.i])
        self.array_with_angular_velocity_real[self.i] = float(self.velocity_angular[self.i])

        self.array_with_linear_velocity_prediction[self.i] = self.predicted_velocity.linear.x
        self.array_with_angular_velocity_prediction[self.i] = self.predicted_velocity.angular.z
        self.get_logger().info('Iteration: "%s"' % self.i) 
        self.i += 1

    def joint_callback(self, msg: JointState):
        """Callback function to process joint state messages."""
        self.joint_names = msg.name
        self.joint_velocities_x[self.i] =msg.velocity[0]
        self.joint_velocities_z[self.i] = msg.velocity[1]


def main(args=None):
    signal.signal(signal.SIGINT, sighandler)
    rclpy.init(args=args)
    global lidar_odom_subscriber
    lidar_odom_subscriber = LidarOdomSubscriber()
   
    rclpy.spin(lidar_odom_subscriber)
    lidar_odom_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()