# Niryo

## Step to Run
1. Git clone this directory into a workspace and catkin build
2. source devel/setup.bash
3. roslaunch niryo_one_moveit_config bed_making_world.launch

## Requirements
```
install pip if you don't have it already

pip install torch

(if above doesn't work, try pip install --no-cache-dir torch)

pip install future
```

Run ```sudo apt-get install ros-kinetic-{}``` for the following:

```
moveit*
rviz*
joint*
ros-control*
control*
trajectory*
gazebo*
```