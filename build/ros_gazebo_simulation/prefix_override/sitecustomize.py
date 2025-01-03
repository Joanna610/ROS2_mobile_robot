import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/joanna/ros2_gz_sim/install/ros_gazebo_simulation'
