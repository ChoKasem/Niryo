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

    