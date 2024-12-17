import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class KeyboardInput(Node):
    def __init__(self):
        super().__init__('keyboard_input')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist = Twist()
        self.get_logger().info("Wprowadź 'w', 's', 'a', 'd' do sterowania.")

    def read_input(self):
        while rclpy.ok():
            key = input()
            if key == 'w':
                self.twist.linear.x = 1.0
            elif key == 's':
                self.twist.linear.x = -1.0
            elif key == 'a':
                self.twist.angular.z = 1.0
            elif key == 'd':
                self.twist.angular.z = -1.0
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0

            self.publisher_.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    keyboard_input = KeyboardInput()
    keyboard_input.read_input()
    keyboard_input.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
