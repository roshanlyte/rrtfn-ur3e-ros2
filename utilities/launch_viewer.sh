#!/bin/bash
source /opt/ros/jazzy/setup.bash

# Generate the URDF from xacro and start robot_state_publisher
# This publishes the robot model but does NOT publish joint_states
# (leaving joint_states free for our RRT*FN planner)

URDF=$(xacro /opt/ros/jazzy/share/ur_description/urdf/ur.urdf.xacro \
  ur_type:=ur3e name:=ur)

ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$URDF" &

sleep 2

# Start RViz
rviz2
