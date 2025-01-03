import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class KeyboardInput(Node):
    def __init__(self):
        super().__init__('keyboard_input')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist = Twist()

    def read_input(self):
        while rclpy.ok():
            key = input()
            if key == 'w':
                self.twist.linear.x = 0.8
                self.twist.angular.z = 0.0
            elif key == 's':
                self.twist.linear.x = self.twist.linear.x - 0.1
                self.twist.angular.z = 0.0
            elif key == 'a':
                self.twist.linear.x = 0.8
                self.twist.angular.z = 0.7
            elif key == 'q':
                self.twist.linear.x = 0.8
                self.twist.angular.z = 0.3
            elif key == 'd':
                self.twist.linear.x = 0.8
                self.twist.angular.z = -0.7
            elif key == 'e':
                self.twist.linear.x = 0.8
                self.twist.angular.z = -0.3
            elif key == 'k':
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0

            self.publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    keyboard_input = KeyboardInput()
    keyboard_input.read_input()
    keyboard_input.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
