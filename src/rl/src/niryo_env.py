#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
import niryo_moveit_commander

# TODO have Niryo inherit moveit commander, or maybe super
class Niryo:
    def __init__(self):
        rospy.loginfo("Initialize Niryo RL Node")
        rospy.init_node('Niryo_RL_Node',
                    anonymous=True)
        self.command = niryo_moveit_commander.MoveGroupPythonInteface()
        self.state = Niryo_State()
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    
    def reset_pose(self):
        # joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.command.go_to_joint_state(-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05)
        

    def compute_reward(self):
        pass

class Niryo_State:
    def __init__(self):
        # rospy.init_node('Niryo_State', anonymous=True)
        rospy.loginfo("Initializing State Subscriber")
        
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
        

def test():
    niryo = Niryo()
    
    # niryo.command.go_to_pose_goal(0.215643591815, 0.117823111836, 0.41643872883, 0.361772034518, 0.609889820653 , 0.606442770113, 0.359697884734)
    niryo.command.go_to_pose_goal(0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344)
    # print("Press Enter")
    # raw_input()
    niryo.reset_pose()


if __name__ == '__main__':
    test()

    
