from arm import Arm, Gripper
from world import World
from base_env import Env

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

    # TODO: modify these to match base_env
    self.action_space.n = 7
    num_joints = 12
    rgb_img_shape = (480, 640, 3)
    depth_img_shape = (480, 640, 1)

    def __init__(self, reward_type = 'dense', distance_threshold = 0.05):

        """Initializes a new Fetch environment.
        Args:
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        # rospy.loginfo("Initialize Niryo RL Node")
        # rospy.init_node('Niryo_RL_Node',
        #             anonymous=True)
        # self.state = State(None,None,None,None)
        self.arm = Arm()            
        self.gripper = Gripper()
        self.world = World()
        self.done = False
        self.info = None

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

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

if __name__ == '__main__':
    print("Inside niryo_env.py")
    niryo = Niryo()
    print(niryo.action_space.n)
