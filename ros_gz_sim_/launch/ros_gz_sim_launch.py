import os
from ament_index_python.packages import get_package_share_directory

from pathlib import Path

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit

from launch_ros.actions import Node

import xacro


def generate_launch_description():

    set_qt_variable = SetEnvironmentVariable('QT_QPA_PLATFORM', 'xcb')

    package_name='ros_gz_sim_' 

    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','rsp_launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    default_world = os.path.join(
        get_package_share_directory(package_name),
        'worlds',
        'empty.world'
        )    

    world = LaunchConfiguration('world')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World to load'
        )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                    launch_arguments={'gz_args': ['-r -v4 ', world], 'on_exit_shutdown': 'true'}.items()
             )
                
    

    spawn_entity = Node(package='ros_gz_sim', executable='create',
                        arguments=['-topic', 'robot_description',
                                   '-name', 'my_bot',
                                   '-z', '0.1'],
                        output='screen')
    
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
        name="diff_drive_spawner"
    )


    bridge_params = os.path.join(get_package_share_directory(package_name),'config','ros_gz_bridge.yaml')
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ]
    )

    # joint_broad_spawner = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=["joint_broad"],
    #     name="joint_broad_spawner"
    # )

    # Launch them all!
    return LaunchDescription([
        set_qt_variable,
        rsp,
        world_arg,
        gazebo,
        spawn_entity,
        # diff_drive_spawner,
        # joint_broad_spawner,
        ros_gz_bridge,
        # twist_mux
    ])