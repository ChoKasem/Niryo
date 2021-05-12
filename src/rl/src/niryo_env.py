#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import GripperCommandActionGoal
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import *
from std_srvs.srv import Empty

import niryo_moveit_commander

# TODO have Niryo inherit moveit commander, or maybe super
class Niryo:
    def __init__(self):
        rospy.loginfo("Initialize Niryo RL Node")
        rospy.init_node('Niryo_RL_Node',
                    anonymous=True)
        self.arm = Arm()            
        self.command = niryo_moveit_commander.MoveGroupPythonInteface("arm")
        
        self.gripper = Gripper()
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    
    def reset_pose(self):
        # starting joint state
        joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.command.go_to_joint_state(joints)
        self.gripper.grab_angle(0)

    def step(self, end_effector_pose, gripper_angle):
        '''
        Step action which move the robot and gripper

        Args: 
            end_effector_pose (list)
            gripper_angle (float)

        Returns:
            observation 
            reward (float)
            done (bool)
            info (string)
        '''
        go_to_pose(end_effector_pose[0],end_effector_pose[1],end_effector_pose[2],end_effector_pose[3],end_effector_pose[4],end_effector_pose[5],end_effector_pose[6])
        gripper.grab_angle(gripper_angle)

        return self.observation, self.reward, self.done, self.info

    def compute_reward(self):
        # possible neg reward if arm hit other object and + reward if get pillow to desire pose
        pass

    def close(self):
        # close the terminal and everything after finish training
        pass

class Arm:
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
        rospy.Subscriber('/')
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
        
class Gripper:
    
    def __init__(self):
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
        self.gripper_pub = rospy.Publisher('/gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)

    def grab_angle(self, angle): #angle max at 1.2
        gripperGoal = GripperCommandActionGoal()
        gripperGoal.goal.command.position = angle
        print(gripperGoal)
        rospy.sleep(1)
        self.gripper_pub.publish(gripperGoal)
        

class World:
    def __init__(self):
        self.pillow_z = self.get_height("Pillow")

    def reset(self):
        # remove bed and pillow and respawn them
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_gazebo_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        ans = self.reset_gazebo_world()

    def get_model_state(self, model):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        state = self.get_model_state(model,"")
        # print(state)
        return state

    def get_height(self, model):
        state = self.get_model_state("Pillow")
        # print(state.pose.position.z)
        return state.pose.position.z

def test_arm():
    # niryo = Niryo()
    niryo.command.go_to_pose_goal(0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344)
    niryo.reset_pose()

def test_gripper():
    # niryo = Niryo()
    rospy.sleep(2)
    niryo.gripper.grab_angle(1.2)
    rospy.sleep(2)
    niryo.gripper.grab_angle(0.3)

def test_world():
    world = World()
    world.reset()

if __name__ == '__main__':
    # niryo = Niryo()
    # test_arm()
    # test_gripper()
    world = World()
    # world.get_model_state("Pillow")
    print(world.pillow_z)
    # world.reset()
    
