import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ros_gazebo_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'description'), glob('description/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joanna',
    maintainer_email='joanna.chrobot610@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_velocity = ros_gazebo_simulation.velocity_publisher:main',
            'data_recorder = ros_gazebo_simulation.data_bag_recorder:main',
            'keyboard_controller = ros_gazebo_simulation.keyboard_controller:main',
            'CNN = ros_gazebo_simulation.CNN_publisher:main',
            'RNN_activate = ros_gazebo_simulation.RNN_velocity_publisher:main',
            'RNN1_activate = ros_gazebo_simulation.RNN_var1_velocity_publisher:main',
            'RNN2_activate = ros_gazebo_simulation.RNN_var2_velocity_publisher:main',
        ],
    },
)
