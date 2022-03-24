#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import GripperCommandActionGoal
from gazebo_msgs.srv import GetModelState, SpawnModel
from geometry_msgs.msg import *
from std_srvs.srv import Empty
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from rl_base import State
import tf


import niryo_moveit_commander

class Niryo:

    """
    From openai gym, The main API methods that users of this class need to know are:
        step
        reset
        render (ignore)
        close
        seed

    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    action_dim = 7
    num_joints = 12
    rgb_img_shape = (480, 640, 3)
    depth_img_shape = (480, 640, 1)

    def __init__(self, reward_type = 'dense', distance_threshold = 0.05):

        """Initializes a new Fetch environment.
        Args:
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        rospy.loginfo("Initialize Niryo RL Node")
        rospy.init_node('Niryo_RL_Node',
                    anonymous=True)
        self.state = State(None,None,None,None)
        self.arm = Arm()            
        self.gripper = Gripper()
        self.world = World()
        self.done = False
        self.info = None

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
    
    # def step(self, step_vector):
    #     """
    #     Pass a [x, y, z, row, pitch, yaw, gripper_angle] vector to go to pose
    #     which will be add delta to it if it's positive or deduct by delta if negative
    #     Note: Angle are in radians

    #     Args:
    #         list of length 7
    #         [x, y, z, row, pitch, yaw, gripper_angle]

    #     return obs, reward, done, info
    #     """
    #     delta = 0.1
    #     pose = self.arm.get_end_effector_pose()
    #     # print(pose)
    #     euler = tf.transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    #     # print(euler)
    #     # print(euler[0] + delta * one_hot_input[3], euler[1] + delta * one_hot_input[4], euler[2] + delta * one_hot_input[5])
    #     q = tf.transformations.quaternion_from_euler(euler[0] + delta * step_vector[3], euler[1] + delta * step_vector[4], euler[2] + delta * step_vector[5])
    #     # print(q)
    #     pose.position.x += delta * step_vector[0]
    #     pose.position.y += delta * step_vector[1]
    #     pose.position.z += delta * step_vector[2]
    #     pose.orientation.x = q[0]
    #     pose.orientation.y = q[1]
    #     pose.orientation.z = q[2]
    #     pose.orientation.w = q[3]

    #     self.go_to_pose(pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    #     self.gripper.grab_angle(self.gripper.gripper_angle + delta * step_vector[6])


    #     return self.get_obs(), self.compute_reward(), self.done, self.info

    def step(self, step_vector):
        """
        Pass a vector of length 14
        [x+, x-, y+, y-, z+, z-, row+, row-, pitch+, pitch-, yaw+, yaw-, gripper+, gripper-] 
        vector to go to pose
        which will be add delta to it if it's positive or deduct by delta if negative
        Note: Angle are in radians

        Args:
            list of length 14
            [x+, x-, y+, y-, z+, z-, row+, row-, pitch+, pitch-, yaw+, yaw-, gripper+, gripper-] 
        

        return obs, reward, done, info
        """
        delta = 0.1 #adjustable distance input
        pose = self.arm.get_end_effector_pose()
        assert step_vector.count(0) == 13 and step_vector.count(1) == 1
        # print(pose)
        euler = tf.transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        q = tf.transformations.quaternion_from_euler(
            euler[0] + delta * (step_vector[6] - step_vector[7]), 
            euler[1] + delta * (step_vector[8] - step_vector[9]),
            euler[2] + delta * (step_vector[10] - step_vector[11])
        )
        # print(q)
        pose.position.x += delta * (step_vector[0] - step_vector[1])
        pose.position.y += delta * (step_vector[2] - step_vector[3])
        pose.position.z += delta * (step_vector[4] - step_vector[5])
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        self.go_to_pose(pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        self.gripper.grab_angle(self.gripper.gripper_angle + delta * (step_vector[12] - step_vector[13]))
        return self.get_obs(), self.compute_reward(), self.done, self.info

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation. The robot will move to dafult position and gazebo world
        is reset.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        
        Returns:
            observation (object): the initial observation.
        """
        self.reset_pose()
        self.world.reset()

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.

        TODO: see how to close gazebo and moveit with python command
        """
        pass
    
    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.arm.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)
    
    def reset_pose(self):
        # starting joint state
        joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.arm.command.go_to_joint_state(joints)
        self.gripper.grab_angle(0)
        return self.get_obs()

    def move(self, end_effector_pose, gripper_angle):
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
        self.get_obs()        
        return self.get_obs(), self.compute_reward(), self.done, self.info

    def compute_reward(self):
        '''
        Compute the Reward, composing of # terms
        1) dist_penalty : penalize distance proportion to distance between goal and pillow
        2) penalized for how off the orientation of pillow is to goal oritation
        3) penalized for getting to deep into the bed (or don't touch bed at all, terminate if it does) or if hit bedframe
        TODO: ?maybe give high reward for placing pillow in correct pose and orientation witin threshold
        '''
        def dist_penalty():
            goal = self.world.pillow_goal_pose
            current = self.world.pillow_pose
            dist = np.sqrt((goal[0] - current.pose.position.x) ** 2 + (goal[1] - current.pose.position.y) ** 2 + (goal[2] - current.pose.position.z) ** 2)
            if self.reward_type == 'sparse':
                return -(dist > self.distance_threshold).astype(np.float32)
            else:
                return -dist
        touch_mattress_penalty = 0
        touch_bedframe_penalty = 0
        end_eff_pose = self.arm.get_end_effector_pose()
        if end_eff_pose.position.z < 0.119:
            touch_matress_penalty = -15
            self.done = True


        if end_eff_pose.position.y > 0.2870:
            touch_bedframe_penalty = -20
            self.done = True
        
        return dist_penalty() + touch_mattress_penalty + touch_bedframe_penalty

    def get_obs(self):
        self.state.rgb = self.arm.image
        self.state.depth = self.arm.depth[..., np.newaxis] # (480,640)->(480,640,1) 
        self.state.joint = np.array(self.arm.joint_angle.position)
        return self.state

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
        self.pillow_goal_pose = [0.4001509859, 0.249076249827, 0.149965204174, -0.000692504034247, 0.00251693882414, 0.999993530658, 0.00247469165438]
        self.pillow_pose = self.get_model_state("Pillow")

    def update_world_state(self):
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
        print(ans)

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

    def spawn(self, model, x, y, z, row, pitch, yaw, random=False):
        '''
        Spawn SDF Model

        Args: 
            model ('pillow' or 'goal'): model that want to spawn wrt to world
                pillow and goal locations are limit to x = [0.25 to 0.4] and y = [-0.15 to 0.2] to keep it within camera and arm workspace 
                z should be 0.19 for pillow and 0.12 for goal
            x, y, z row, pitch, yaw : coordinate and rotation with repect to the world coordinate
            
        Returns:
            None
        '''
        
        if model.lower() == "pillow":
            f = open('/home/joker/Niryo/src/niryo_one_ros_simulation/niryo_one_gazebo/models/pillow/model.sdf','r')
        elif model.lower() == "goal":
            f = open('/home/joker/Niryo/src/niryo_one_ros_simulation/niryo_one_gazebo/models/goal/model.sdf','r')
        #if goal
        initial_pose = Pose()
        q = tf.transformations.quaternion_from_euler(row, pitch, yaw)
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z
        initial_pose.orientation.x = q[0]
        initial_pose.orientation.y = q[1]
        initial_pose.orientation.z = q[2]
        initial_pose.orientation.w = q[3]
        sdff = f.read()

        # if random==True:
            # TODO Spawn the goal block and pillow randomly?

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox("pillow", sdff, "", initial_pose, "world")

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
    print(niryo.world.get_model_state("Goal"))
    print("Get Height")
    print(niryo.world.get_height("Pillow"))
    niryo.world.reset()

def test_Niryo():
    print('Test Step')
    # end_effector_pose = [0.350840341432, -0.058138712168, 0.276432223498, 0.50174247142, 0.501506407284, 0.498433947182, 0.498306548344]
    #gripper_angle = 0.6
    # niryo.move(end_effector_pose,gripper_angle)
    # print("Printing Observation")
    # print(niryo.get_obs())
    one_input = [0,0,-1,0,0,0,0]
    niryo.step(one_input)

def test_reward():
    print('Test Reward')
    print(niryo.compute_reward())

def test_rl_process():
    one_input = np.identity(14)
    for i in range(14):
        niryo.step(one_input[i].tolist())
    niryo.reset()
    


if __name__ == '__main__':
    niryo = Niryo()
    # test_arm()
    # raw_input()
    # test_gripper()
    # raw_input()
    # test_world()
    # raw_input()
    # test_Niryo()
    # print(niryo.arm.image)
    # print(niryo.get_obs())
    # test_reward()
    # niryo.world.spawn("Pillow", 0.4, -0.15, .2, 0 ,0,0)
    test_rl_process()
    # niryo.world.reset()
    print("Main Done")
    