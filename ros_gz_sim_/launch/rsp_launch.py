import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

import xacro


def generate_launch_description():

    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('ros_gz_sim_'))
    xacro_file = os.path.join(pkg_path,'description','robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    
    # Create a robot_state_publisher node
    params = {'robot_description': robot_description_config.toxml(), 'use_sim_time': use_sim_time}
    node_robot_state_publisher = Node(
    
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params],
        name="node_robot_state_publisher"
    )

    


    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'),

        node_robot_state_publisher
    ])

    # use_sim_time = LaunchConfiguration('use_sim_time')

    # Ścieżka do pliku Xacro
    # xacro_file = os.path.join(
    #     get_package_share_directory('ros_gz_sim_'),
    #     'description',
    #     'robot.urdf.xacro'
    # )

    # # Konwertowanie Xacro na URDF za pomocą komendy
    # robot_description = Command(['xacro ', xacro_file])

    # # Uruchomienie node'a robot_state_publisher, aby publikować opis robota
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     output='screen',
    #     parameters=[{
    #         'robot_description': robot_description
    #     }]
    # )

    # urdf_file = '~/new_project_new_me/src/ros_gz_sim_/description/model.urdf'

    # generate_urdf = ExecuteProcess(
    #     cmd=['xacro', '~/new_project_new_me/src/ros_gz_sim_/description/robot.urdf.xacro', '>', urdf_file],
    #     shell=True,
    #     output='screen'
    # )    

    # Uruchomienie symulatora Gazebo z pustym światem
    # gz_sim_process = ExecuteProcess(
    #     cmd=['QT_QPA_PLATFORM=xcb gz sim empty.sdf'],
    #     shell=True,
    #     output='screen'
    # )

    # Dodanie robota do Gazebo
    # spawn_robot_process = ExecuteProcess(
    #     cmd=[
    #         'QT_QPA_PLATFORM=xcb gz service -s /world/empty/create',
    #         '--reqtype gz.msgs.EntityFactory',
    #         '--reptype gz.msgs.Boolean',
    #         '--timeout 1000',
    #         f'--req \'sdf_filename: "/home/joanna/new_project_new_me/src/ros_gz_sim_/description/model.urdf", name: "robot"\''
    #     ],
    #     shell=True,
    #     output='screen'
    # )

    # return LaunchDescription([
    #     gz_sim_process,         # Uruchomienie Gazebo
    #     robot_state_publisher_node,  # Publikacja opisu robota
    #     spawn_robot_process     # Zrespawnienie robota
    # ])