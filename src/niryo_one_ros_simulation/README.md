# Niryo One ROS Simulation
Licensed under GPLv3 (see [LICENSE file](https://github.com/NiryoRobotics/niryo_one_ros_simulation/blob/master/LICENSE))

Works on ROS Kinetic/Melodic.

ROS simulation for the robot [Niryo One](https://niryo.com/niryo-one/). You can control the robot using ros_control, Moveit, and see a 3D simulation on both Rviz and Gazebo.

## Install from source

Get the code:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/NiryoRobotics/niryo_one_ros_simulation.git .
```

Build the packages:

```bash
cd ~/catkin_ws
catkin_make
```

Don't forget to use those commands before you try to launch anything (you can add them in your .bashrc) :

```bash
source /opt/ros/melodic/setup.bash # replace 'melodic' by your ROS version
source ~/catkin_ws/devel/setup.bash
```

## Display Niryo One with Rviz

To simply display the robot and get to move each joint separately, run:

```bash
roslaunch niryo_one_description display.launch
```

## Moveit demo

The Moveit demo will start Niryo One in Rviz and provide motion planning functionalities (with a 'fake' controller):

```bash
roslaunch niryo_one_moveit_config demo.launch
```

## Start the Gazebo simulation

Developed and tested on ROS Melodic/Gazebo 9.

First start Gazebo (empty world) and Niryo One model:

```bash
roslaunch niryo_one_gazebo niryo_one_world.launch
```

Then, start the controllers (ros_control):

```bash
roslaunch niryo_one_gazebo niryo_one_control.launch
```

The ROS interface to control Niryo One is the same for the [real robot](https://github.com/NiryoRobotics/niryo_one_ros) and the Gazebo simulation.

The controller used for Niryo One is a joint\_trajectory\_controller (from ros\_control). See the [joint\_trajectory\_controller documentation](http://wiki.ros.org/joint_trajectory_controller) to know how to use it.

If you want to add motion planning with Moveit, also run (after the controllers):

```bash
roslaunch niryo_one_moveit_config move_group.launch
```

You can now use the Python or C++ MoveGroup interface. This interface will call the Moveit motion planning functionality and then send the computed plan to the joint\_trajectory\_controller.
