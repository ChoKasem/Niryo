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
        self.state = Niryo_State()            
        self.command = niryo_moveit_commander.MoveGroupPythonInteface("arm")
        
        # gripper have move group, but don't have planning tool so cannot control
        self.gripper_cmd = niryo_moveit_commander.MoveGroupPythonInteface("gripper")
        self.gripper = Gripper_State()
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    
    def reset_pose(self):
        # starting joint state
        joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.command.go_to_joint_state(joints)
        

    def compute_reward(self):
        pass

class Niryo_State:
    def __init__(self):
        '''
        0 = joint_1
        1 = joint_2
        2 = joint_3
        3 = joint_4
        4 = joint_5
        5 = joint_6
        6 = tool_joint // cannot be control here, just for connecting with tools
        '''
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
        
class Gripper_State:
    '''
    0 = tool_joint
    1 = mainsupport_joint
    2 = gripper_joint
    3 = left_clamp_joint
    4 = left_rod_joint
    5 = motor_joint
    6 = right_gear_joint
    7 = right_clamp_joint
    8 = right_rod_joint
    '''
    pass

def test():
    niryo = Niryo()
    tool_joint = [3.41760362232e-08, -9.71873266309e-07, 2.86864792365e-07, 0.0134953835178, -0.0164044312931, 0.0168274248518]
    niryo.gripper_cmd.go_to_joint_state(tool_joint)
    niryo.command.go_to_pose_goal(0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344)
    print("Press Enter")
    raw_input()
    niryo.reset_pose()

def test2():
    niryo = Niryo()
    while True:
        print(niryo.state.joint_angle)

if __name__ == '__main__':
    test2()
    
