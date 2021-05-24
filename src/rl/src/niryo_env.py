#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import GripperCommandActionGoal
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import *
from std_srvs.srv import Empty
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

import niryo_moveit_commander

class Niryo:
    def __init__(self):
        rospy.loginfo("Initialize Niryo RL Node")
        rospy.init_node('Niryo_RL_Node',
                    anonymous=True)
        self.arm = Arm()            
        self.gripper = Gripper()
        self.world = World()
        self.done = False
        self.info = None
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.arm.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    
    def reset_pose(self):
        # starting joint state
        joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.arm.command.go_to_joint_state(joints)
        self.gripper.grab_angle(0)

    def step(self, end_effector_pose, gripper_angle):
        '''
        Step action which move the robot and gripper

        Args: 
            end_effector_pose (list[7])
            gripper_angle (float)

        Returns:
            observation : end_effector_pose (list[7]), joint_state (list), rgb, depth, gripper_angle
            reward (float)
            done (bool) : tell if pillow is at goal pose
            info (string)
        '''
        self.go_to_pose(end_effector_pose[0],end_effector_pose[1],end_effector_pose[2],end_effector_pose[3],end_effector_pose[4],end_effector_pose[5],end_effector_pose[6])
        self.gripper.grab_angle(gripper_angle)
        _, _ , img, depth, _ = self.get_obs()
        # print(img.data)
        print(img.shape)
        cv2.imshow("image", img)
        cv2.imshow("depth", depth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.imshow(img.data)
        
        return self.get_obs(), self.compute_reward(), self.done, self.info

    def compute_reward(self):
        # possible neg reward if arm hit other object and + reward if get pillow to desire pose
        # include z orientation
        if self.world.pillow_move() is True:
            return self.loss()
        
        return 0

    def loss(self):
        desire = self.world.pillow_desire_pose
        current = self.world.pillow_pose
        dist = np.sqrt((desire[0] - current.pose.position.x) ** 2 + (desire[1] - current.pose.position.y) ** 2 + (desire[2] - current.pose.position.z) ** 2)
        L = 100
        reward = L / (1 + np.exp(dist)) 
        return reward
         

    def get_obs(self):
        return self.arm.get_end_effector_pose(), self.arm.joint_angle, self.arm.image, self.arm.depth, self.gripper.gripper_angle

    
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
        self.image = None
        self.depth = None
        self.joint_angle = None
        self.bridge = CvBridge()
        self.command = niryo_moveit_commander.MoveGroupPythonInteface("arm")
        
        # rospy.init_node('Niryo_State', anonymous=True)
        rospy.loginfo("Initializing State Subscriber")
        
        rospy.Subscriber('/joint_states', JointState, self.joint_cb)
        # rospy.Subscriber('/')
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_cb)
        rospy.sleep(1)

    def get_state_callback(self, data):
        self.joint_angle

    def image_cb(self, msg):
        
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def joint_cb(self, msg):
        self.joint_angle = msg

    def get_end_effector_pose(self):
        return self.command.get_pose()
        
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
        self.gripper_angle = 0
        self.gripper_pub = rospy.Publisher('/gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=10)
        rospy.sleep(1)

    def grab_angle(self, angle): #angle max at 1.2
        if angle > 1.2:
            angle = 1.2
        gripperGoal = GripperCommandActionGoal()
        gripperGoal.goal.command.position = angle
        print(gripperGoal)
        rospy.sleep(1)
        self.gripper_angle = angle
        self.gripper_pub.publish(gripperGoal)
        

class World:
    def __init__(self):
        rospy.sleep(1)
        # self.pillow_z = self.get_height("Pillow")
        self.pillow_desire_pose = [0.4001509859, 0.249076249827, 0.149965204174, -0.000692504034247, 0.00251693882414, 0.999993530658, 0.00247469165438]
        self.pillow_pose = self.get_model_state("Pillow")

    def pillow_move(self):
        # if pillow doesn't move, return 0
        # if pillow move, calculate reward
        new_pose = self.get_model_state("Pillow")
        if new_pose == self.pillow_pose:
            return False

        else:
            self.pillow_pose = new_pose
            return True

    def pillow_move_up(self):
        if np.abs(self.get_height("Pillow") - self.pillow_z) > 0.01:
            return True
        return False
    
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
            get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        except rospy.ServiceException as e:
            print("Error")
            print("Service call failed: %s"%e)
        state = get_state(model, '')
        # print(state)
        return state

    def get_height(self, model):
        state = self.get_model_state(model)
        # print(state.pose.position.z)
        return state.pose.position.z

def test_arm():
    # niryo = Niryo()
    niryo.arm.command.go_to_pose_goal(0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344)
    print("Get Pose")
    print(niryo.arm.get_end_effector_pose())
    print('Reseting')
    niryo.reset_pose()
    # print('going to final target')
    # niryo.command.go_to_pose_goal(0.245375560498, 8.29996046947e-08, 0.417146039267, 0.500484688157, 0.500476344648, 0.499523261308, 0.499514781343)

def test_gripper():
    rospy.sleep(2)
    niryo.gripper.grab_angle(1.2)
    rospy.sleep(2)
    niryo.gripper.grab_angle(0.3)

def test_world():
    print("Get model state")
    print(niryo.world.get_model_state("Pillow"))
    print("Get Height")
    print(niryo.world.get_height("Pillow"))
    niryo.world.reset()

def test_Niryo():
    print('Test Step')
    end_effector_pose = [0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344]
    gripper_angle = 0.6
    niryo.step(end_effector_pose,gripper_angle)
    # print("Printing Observation")
    # print(niryo.get_obs())

if __name__ == '__main__':
    niryo = Niryo()
    # test_arm()
    # raw_input()
    # test_gripper()
    # raw_input()
    # test_world()
    # raw_input()
    test_Niryo()
    print("Done")
    
