import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_vel)
        self.i_x = 0.0
        self.i_z_a = 1.0
        self.time = 0
        self.dol = True
        self.gora = False

    def publish_vel(self):
        # Wyświetlanie prędkości liniowych i kątowych
        vel = Twist()
        vel.linear.x = self.i_x
        vel.linear.y = 0.0
        vel.linear.z = 0.0
        vel.angular.x = 0.0 
        vel.angular.y = 0.0
        vel.angular.z = self.i_z_a      

        self.publisher_.publish(vel)
        self.get_logger().info(f'Publishing: {vel}')

        self.i_x = self.i_x + 0.05
        self.i_z_a = self.i_z_a + 0.02

        if self.i_x > 1:
            vel.linear.x = 0.0
            vel.angular.z = 0.0
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    vel_publisher = VelocityPublisher()
    rclpy.spin(vel_publisher)
    vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()