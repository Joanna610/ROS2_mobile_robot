import os
import time
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
# from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import csv
import signal
import sys
from rclpy.serialization import deserialize_message
import rosbag2_py

writer = None
date = None

def sighandler(signum, frame):
    print("Odebrano sygnał:", signum)
    global writer
    global date
    if signum == 2:
        writer.close()
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri='bag_folder'+date, storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        # Pobranie listy topiców
        topics = reader.get_all_topics_and_types()

        # Bufory dla danych z różnych topiców
        odom_data = {'odom_linear_x': None, 'odom_angular_z': None}
        scan_data = {'ranges': None}


        file_csv = os.path.join('bag_folder'+date, 'rosbag_data_' + date + '.csv')
        # Otwieranie pliku CSV do zapisu
        with open(file_csv, mode='w', newline='') as csv_file:
            fieldnames = ['odom_linear_x', 'odom_angular_z'] + [f'range_{i+1}' for i in range(360)]
            writer_ = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer_.writeheader()

            # Iteracja po wiadomościach
            while reader.has_next():
                topic, data, t = reader.read_next()

                if topic == '/odom':
                    # Deserializacja wiadomości typu Odometry
                    odom_msg = deserialize_message(data, Odometry)
                                    
                    # Aktualizacja bufora odom
                    odom_data['odom_linear_x'] = odom_msg.twist.twist.linear.x
                    odom_data['odom_angular_z'] = odom_msg.twist.twist.angular.z

                elif topic == '/scan':
                    # Deserializacja wiadomości typu LaserScan
                    scan_msg = deserialize_message(data, LaserScan)
                    
                    # Aktualizacja bufora scan
                    # scan_data['ranges'] = scan_msg.ranges[:360]
                    row = {'odom_linear_x': odom_data['odom_linear_x'], 'odom_angular_z': odom_data['odom_angular_z']}
                    ranges = list(scan_msg.ranges[:360]) + [None] * (360 - len(scan_msg.ranges))
                    row.update({f'range_{i+1}': ranges[i] for i in range(360)})

                    # Zapisz wiersz do pliku
                    writer_.writerow(row)
    sys.exit(0)  





class SimpleBagRecorder(Node):
    def __init__(self):
        super().__init__('simple_bag_recorder')
        global writer
        global date
        writer = rosbag2_py.SequentialWriter()

        date = time.strftime('%Y%m%d_%H%M%S')
        storage_options = rosbag2_py._storage.StorageOptions(
            uri='bag_folder'+date,
            storage_id='mcap')

        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        writer.open(storage_options, converter_options)
        
        topic_odom_info = rosbag2_py._storage.TopicMetadata(
            id=1,
            name='/odom',
            type='nav_msgs/msg/Odometry',
            serialization_format='cdr')
        writer.create_topic(topic_odom_info)

        topic_lidar_info = rosbag2_py._storage.TopicMetadata(
            id=2,
            name='/scan',
            type='sensor_msgs/msg/LaserScan',
            serialization_format='cdr')
        writer.create_topic(topic_lidar_info)

        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)     

    def odom_callback(self, msg):
        writer.write(
            '/odom',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)
        
    def scan_callback(self, msg):
        # Zapis danych z /scan do pliku
        writer.write(
            '/scan',
            serialize_message(msg),
            self.get_clock().now().nanoseconds)
        
def main(args=None):
    signal.signal(signal.SIGINT, sighandler)
    rclpy.init(args=args)
    sbr = SimpleBagRecorder()
    rclpy.spin(sbr)
    rclpy.shutdown()


if __name__ == '__main__':
    main()