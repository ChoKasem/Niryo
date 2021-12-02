from abc import abstractmethod
import numpy as np
import torch

class ActionSpace(object):
    def __init__(self):
        self.n = 0
        self.action = []
    def sample(self):
        num = np.random.randint(low = 0, high=self.n)
        print("action num: ", num)
        return self.action[num]

class ObservationSpace(object):
    def __init__(self, rgb, pillow, goal, joint, gripper):
        img = torch.from_numpy(rgb).float()
        img = torch.reshape(img, (-1, 3, 480, 640))
        self.rgb = img
        self.pillow = torch.reshape(torch.tensor(self.pose2vector(pillow)), (-1,7))
        self.goal = torch.reshape(torch.tensor(self.pose2vector(goal)), (-1, 7))
        self.joint = torch.reshape(torch.tensor(joint), (-1,6))
        self.gripper = torch.reshape(torch.tensor([gripper]).float(), (-1,1))
    
    def pose2vector(self, pose):
        # print(pose)
        vec = [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return vec

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """

    # Set this in SOME subclasses
    # metadata = {"render.modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    # Set these in ALL subclasses
    def __init__(self):
        self.action_space = ActionSpace()
        self.observation_space = None

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError



if __name__ == '__main__':
    print("Inside base_env.py")