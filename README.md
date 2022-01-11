# Niryo

## Step to Run
1. Git clone this repo into the home folder (it will create a Niryo workspace)
2. cd into the Niryo folder and catkin build
3. type source devel/setup.bash into terminal
4. roslaunch niryo_one_moveit_config bed_making_world.launch

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