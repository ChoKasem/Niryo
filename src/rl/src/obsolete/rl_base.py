import numpy as np
import torch

class State:
    def __init__(self, rgb, depth, joint, pillow):
        self.rgb = rgb 
        self.depth = depth
        self.joint = joint
        self.pillow = pillow
    
    def to_tensor(self):
        # Load into tensors
        return State(
            rgb=torch.FloatTensor(self.rgb).unsqueeze(0),
            depth=torch.FloatTensor(self.depth).unsqueeze(0),
            joint=torch.FloatTensor(self.joint),
            pillow=torch.FloatTensor(self.pillow)
        )
    
    def transform(self, func):
        return State(
            rgb=func(self.rgb),
            depth=func(self.depth),
            joint=func(self.joint),
            pillow=func(self.pillow)
        ) 

    @staticmethod
    def to_tensor_object(state_list):
        rgb, depth, joint, pillow = [], [], [], []
        for state in state_list:
            rgb.append(torch.FloatTensor(state.rgb))
            depth.append(torch.FloatTensor(state.depth))
            joint.append(torch.FloatTensor(state.joint))
            pillow.append(torch.FloatTensor(state.pillow))
        return State(rgb, depth, joint, pillow)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def get_sample_state(img_width, img_height, num_joints):
    # Use some sample values
    rgb_img = np.random.uniform(0, 255, (img_width, img_height, 3))
    depth_img = np.random.uniform(0, 1, (img_width, img_height, 1))
    joint_angles = np.random.uniform(0, np.pi, (num_joints))
    pillow_pose = np.random.uniform(0, 100, (3))

    return State(rgb_img, depth_img, joint_angles, pillow_pose)
