#!/usr/bin/env python3

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    # Define a launch configuration for simulation time
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Return the LaunchDescription with all actions
    return LaunchDescription([
        # Declare a launch argument for simulation time
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        # Include the RViz launch file with custom configuration
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
            ),
            launch_arguments={
                "config": PathJoinSubstitution([
                    FindPackageShare("autonomy_repo"),
                    "rviz",
                    "default.rviz",
                ]),
                "use_sim_time": use_sim_time,
            }.items(),
        ),

        # Relay RViz goal pose to a specific channel
        Node(
            package="asl_tb3_lib",
            executable="rviz_goal_relay.py",
            parameters=[{"output_channel": "/cmd_nav"}],
        ),

        # State publisher for TurtleBot
        Node(
            package="asl_tb3_lib",
            executable="state_publisher.py",
        ),

        # Student's heading controller node
        Node(
            package="autonomy_repo",
            executable="navigator.py",
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        # Student's explorer node with a delay
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="autonomy_repo",
                    executable="explorer.py",  # Changed the name here
                    parameters=[{"use_sim_time": use_sim_time}],
                )
            ],
        ),
    ])
