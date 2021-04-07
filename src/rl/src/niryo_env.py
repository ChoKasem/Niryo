#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
import niryo_moveit_commander

# TODO have Niryo inherit moveit commander, or maybe super
class Niryo:
    def __init__(self):
        self.state = Niryo_State()
        self.command = niryo_moveit_commander.MoveGroupPythonInteface()
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    

    def compute_reward(self):
        pass

class Niryo_State:
    def __init__(self):
        rospy.init_node('Niryo_State', anonymous=True)
        rospy.loginfo("Initializing Subscriber")
        
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)

        #need to get pillow state from Gazebo (only for simulation, for real robot would be image processing)

        self.image = None
        self.depth = None
        self.joint_angle = None
    
    def get_state_callback(self, data):
        self.joint_angle

    def image_cb(self, msg):
        self.image = msg
    
    def depth_cb(self, msg):
        self.depth = msg

    def joint_cb(self, msg):
        self.joint_angle = msg
        


if __name__ == '__main__':
    # state = Niryo_State()
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     print(state.depth)
    #     rospy.sleep(5)
    
    # niryo_moveit_commander.main()

    niryo = Niryo()
