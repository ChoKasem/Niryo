import rospy
import numpy as np
import tf
import torch

from arm import Arm, Gripper, PlanningState
from world import World
from base_env import Env, ActionSpace, ObservationSpace

class Niryo(Env):

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

    num_joints = 12
    rgb_img_shape = (480, 640, 3)
    depth_img_shape = (480, 640, 1)

    def __init__(self, reward_type = 'dense', distance_threshold = 0.05, randomspawn = False):

        """Initializes a new Fetch environment.
        Args:
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        # rospy.loginfo("Initialize Niryo RL Node")
        # rospy.init_node('Niryo_RL_Node',
        #             anonymous=True)
        # self.state = State(None,None,None,None)
        super(Niryo, self).__init__()
        self.action_space.n = 14
        self.action_space.action = np.identity(14)

        # parts of robot
        self.arm = Arm()            
        self.gripper = Gripper()
        self.world = World()
        self.planningstate = PlanningState()
        self.done = False
        self.info = None
        self.randomspawn = randomspawn


        self.reward_type = reward_type.lower()
        self.distance_threshold = distance_threshold

        print("Niryo Env created: Reward Type = ", self.reward_type, ", Distance Threshold = ", self.distance_threshold, ", Random Spawn = ", self.randomspawn)

    def step(self, step_vector):
        """
        Pass a vector of length 14 or an integer between 0 and 13
        [x+, x-, y+, y-, z+, z-, row+, row-, pitch+, pitch-, yaw+, yaw-, gripper+, gripper-] 
        vector to go to pose
        which will be add delta to it if it's positive or deduct by delta if negative
        Note: Angle are in radians

        Args:
            list of length 14
            [x+, x-, y+, y-, z+, z-, row+, row-, pitch+, pitch-, yaw+, yaw-, gripper+, gripper-] 
        

        return obs, reward, done, info
        """
        if isinstance(step_vector, int):
            step_vector =  self.action_space.action[step_vector]
        delta = 0.05 #adjustable distance input
        delta_angle = 0.3 # could have another number for angle
        delta_gripper = 1.2 # have number for gripper
        pose = self.arm.get_end_effector_pose()
        # assert step_vector.count(0) == 13 and step_vector.count(1) == 1
        # print(pose)
        euler = tf.transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        q = tf.transformations.quaternion_from_euler(
            euler[0] + delta_angle * (step_vector[6] - step_vector[7]), 
            euler[1] + delta_angle * (step_vector[8] - step_vector[9]),
            euler[2] + delta_angle * (step_vector[10] - step_vector[11])
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
        self.gripper.grab_angle(self.gripper.gripper_angle + delta_gripper * (step_vector[12] - step_vector[13]))
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
        self.world.reset(random=self.randomspawn)
        return self.get_obs()

    def reset_pose(self):
        # starting joint state
        joints = [-4.00038318737e-05, -0.00169649498877, -0.00135103272703, 1.82992589703e-05, -0.0005746965517, 7.78535278902e-05]
        self.arm.command.go_to_joint_state(joints)
        self.gripper.grab_angle(0)
        return self.get_obs()

    def go_to_pose(self, pos_x = 0, pos_y= 0, pos_z = 0, ori_x = 0, ori_y = 0, ori_z = 0, ori_w = 0):
        self.arm.command.go_to_pose_goal(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w)

    def compute_reward(self):
        '''
        Compute the Reward, composing of 4 terms
        1) dist_penalty : penalize distance proportion to distance between goal and pillow
        TODO: penalized for how off the orientation of pillow is to goal oritation
        TODO: penalized for getting to deep into the bed (or don't touch bed at all, terminate if it does) or if hit bedframe
        TODO: ?maybe give high reward for placing pillow in correct pose and orientation witin threshold
        : check for error in step, if can't find that path, should give negative reward
        '''
        self.world.update_world_state()
        
        def dist_reward():
            goal = self.world.goal_pose
            current = self.world.pillow_pose
            sparse_reward_amount = 100
            scale_factor = 100
            # print("Goal")
            # print(goal)
            # print("Current")
            # print(current)

        # Main Reward and Penalty
        # Reward for moving the pillow witihin threshold distance of the goal
        # Penalty otherwise proportion the the distance between pillow and goal
            # reward (ie penalty) according to the distance between pillow and goal
            dist = np.sqrt((goal.pose.position.x - current.pose.position.x) ** 2 + (goal.pose.position.y - current.pose.position.y) ** 2 + (goal.pose.position.z - current.pose.position.z) ** 2)
            if self.reward_type == 'sparse':
                # sparse reward give huge reward when episode end 
                print("Sparse Reward Calc")
                if self.done == True:
                    if dist < distance_threshold:
                        return sparse_reward_amount
                    else:
                        return -sparse_reward_amount -dist*scale_factor
                else:
                    return 0
            else:
                # dense reward give feedback from every action
                print("Dense Reward Calc")
                return -dist*scale_factor
        touch_mattress_penalty = 0
        touch_bedframe_penalty = 0
        end_eff_pose = self.arm.get_end_effector_pose()
        
        # Penalty for touching the mattrss (by going lower than specify height)
        if end_eff_pose.position.z < 0.15: #0.159 if with bedframe, but shouldn't be much different and depend on bedframe model (just run arm.py to check end effector pose)
            touch_matress_penalty = -150
            self.done = True

        # Penalty for trying to move to far and touch bedframe (going toward the left too much)
        if end_eff_pose.position.y > 0.2870:
            touch_bedframe_penalty = -20
            self.done = True
        
        # Penalty for trying to go outside workspace of robot
        path_planning_penalty = 0
        if self.planningstate.state_status != 3: #if no solution is found
            print(self.planningstate.state_text)
            path_planning_penalty = -1000
        
        total_reward = dist_reward() + touch_mattress_penalty + touch_bedframe_penalty + path_planning_penalty
        print("Total Reward is: ", total_reward)
        return total_reward

    def get_obs(self):
        state = ObservationSpace(self.arm.image, self.world.pillow_pose.pose, self.world.goal_pose.pose, self.arm.joint_angle.position[:6], self.gripper.gripper_angle)
        return state

if __name__ == '__main__':
    print("Inside niryo_env.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('Niryo_Env_Test_Node',
                    anonymous=True)
    # niryo = Niryo()
    # print(niryo.action_space.sample())
    # print("ready to move")
    # niryo.step(niryo.action_space.action[6])
    # niryo.step(niryo.action_space.action[7])
    # niryo.step(niryo.action_space.action[8])
    # niryo.step(niryo.action_space.action[9])
    # niryo.step(niryo.action_space.action[10])
    # niryo.step(niryo.action_space.action[11])
    # niryo.step(niryo.action_space.action[12])
    # niryo.step(niryo.action_space.action[12])
    # niryo.step(niryo.action_space.action[13])

    # print("computing reward")
    # print(niryo.compute_reward())
    # print("Done")

    # print("getting observation")
    # obs = niryo.get_obs()
    # print(obs.rgb)
    # print(obs.pillow)
    # print(obs.goal)
    # print(obs.joint)
    # print(obs.gripper)

    niryo = Niryo(randomspawn=True)
    print("ready to move")
    # niryo.reset()
    niryo.step(niryo.action_space.action[0])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    raw_input()
    niryo.step(niryo.action_space.action[1])
    # niryo.step(niryo.action_space.sample())
    
