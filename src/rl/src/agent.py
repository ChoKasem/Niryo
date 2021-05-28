"""Definitions of the ActorCritic agent.

Acronyms:
Input shape => IS
Output shape => OS
"""

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn

from agent_helpers import init_weights, get_cu_OS, ConvUnit, ConvUnitPars

class ActorCritic(nn.Module):
    """Details
    State: RGB Image + Depth Image + Joint Angles + Pillow Pose
    Output: Joint Angles
    """

    def __init__(self, rgb_img_shape, depth_img_shape, num_joints, pillow_pose_size, num_outputs):
        """Initializing the Actor Critic neural networks.
        Two neural nets: 1. Actor, 2. Critic, both of which have
        same architecture. Each neural net has a portion for each state
        component as well as a portion for combining them all.
        ^The portions make up a full neural network.
        """
        assert len(rgb_img_shape) == 3 and len(depth_img_shape) == 3
        super(ActorCritic, self).__init__()

        self.rgb_img_shape = rgb_img_shape
        self.depth_img_shape = depth_img_shape
        self.num_joints = num_joints
        self.pillow_pose_size = pillow_pose_size

        self.rgb_OS = None
        self.depth_OS = None
        self.pillow_OS = None
        self.joint_OS = None
        self.img_combine_OS = None
        self.full_combine_OS = 32

        self.actor_rgb = self.get_rgb_module()
        self.actor_depth = self.get_depth_module()
        self.actor_img_combine = self.get_img_combine_module()
        self.actor_joint = self.get_joint_module()
        self.actor_pillow = self.get_pillow_module()
        self.actor_pillow_joint_combine = self.get_pillow_joint_combine_module()
        self.actor_full_combine = self.get_full_combine_module()
        self.actor_output = nn.Sequential(
            nn.Linear(self.full_combine_OS, num_outputs),
            nn.LogSoftmax(1)
        )

        self.critic_rgb = self.get_rgb_module()
        self.critic_depth = self.get_depth_module()
        self.critic_img_combine = self.get_img_combine_module()
        self.critic_joint = self.get_joint_module()
        self.critic_pillow = self.get_pillow_module()
        self.critic_pillow_joint_combine = self.get_pillow_joint_combine_module()
        self.critic_full_combine = self.get_full_combine_module()
        self.critic_output = nn.Sequential(
            nn.Linear(self.full_combine_OS, 1)
        )
        
        self.apply(init_weights)

    def forward(self, state):
        """Performs forward pass for the two NNs.
        Importantly, the input reshaping is done here.

        Args for the state object:
            rgb (tensor): Batched 3D image.
                4D tensor (Batch Size, Height, Width, Number Channels)
                Height, width, and number of channels much match the input
                shape given in the initialization of the neural net
            depth (tensor): Batched ??? How many dims?
            joint (tensor): Batched 6D tensor containing joint angles
            pillow (tensor): Batched _D tensor containing pillow pose

        Returns:
            (Categorical, tensor): Action distribution and the value of
                the state
        """
        # Image reshaping
        try:
            batch_size, length, width, num_channels = state.rgb.shape
        except:
            print(state.rgb.shape)
            raise
        rgb_reshape = state.rgb.view(batch_size, num_channels, length, width)
        batch_size, length, width, num_channels = state.depth.shape
        depth_reshape = state.depth.view(batch_size, num_channels, length, width)

        # Actor Forward Pass
        rgb_portion = self.actor_rgb(rgb_reshape)
        depth_portion = self.actor_depth(depth_reshape)
        joint_portion = self.actor_joint(state.joint)
        pillow_portion = self.actor_pillow(state.pillow)
        img_portion = self.actor_img_combine(
            # Add to combine
            rgb_portion + depth_portion
        )
        pillow_joint_portion = self.actor_pillow_joint_combine(
            # Concatenate to combine
            torch.cat((pillow_portion, joint_portion), dim=-1)
        )
        full_combo = self.actor_full_combine(
            # Add to combine
            img_portion + pillow_joint_portion
        )
        action_probs = self.actor_output(full_combo)
        action_dist = Categorical(action_probs)

        # Critic Forward Pass
        rgb_portion = self.critic_rgb(rgb_reshape)
        depth_portion = self.critic_depth(depth_reshape)
        joint_portion = self.critic_joint(state.joint)
        pillow_portion = self.critic_pillow(state.pillow)
        img_portion = self.critic_img_combine(rgb_portion + depth_portion)
        pillow_joint_portion = self.critic_pillow_joint_combine(
            torch.cat((pillow_portion, joint_portion), dim=-1)
        )
        full_combo = self.critic_full_combine(
            img_portion + pillow_joint_portion)
        value = self.critic_output(full_combo)

        # Returning action distribution and value
        return action_dist, value

    def get_rgb_module(self):
        # RGB Image Module ----------------------------------------------------
        rgb1_pars = ConvUnitPars(16, 3, 1, 2, 4, 2)
        rgb2_pars = ConvUnitPars(16, 2, 1, 2, 4, 2)
        print("---------------\nRGB layers output shapes")
        rgb1_OS = get_cu_OS(self.rgb_img_shape, rgb1_pars)
        self.rgb_OS = get_cu_OS(rgb1_OS, rgb2_pars)
        print(self.rgb_img_shape, rgb1_OS, self.rgb_OS)
        print("---------------")
        return nn.Sequential(
            ConvUnit(
                self.rgb_img_shape,
                rgb1_pars,
            ),
            ConvUnit(
                rgb1_OS,
                rgb2_pars,
            ),
        )

    def get_depth_module(self):
        # Depth Image Module --------------------------------------------------
        depth1_pars = ConvUnitPars(16, 3, 1, 2, 4, 2)
        depth2_pars = ConvUnitPars(16, 2, 1, 2, 4, 2)
        print("---------------\nDepth layers output shapes")
        depth1_OS = get_cu_OS(
            self.depth_img_shape, depth1_pars)
        self.depth_OS = get_cu_OS(
            depth1_OS, depth2_pars)
        print(self.depth_img_shape, depth1_OS, self.depth_OS)
        print("---------------")
        return nn.Sequential(
            ConvUnit(
                self.depth_img_shape,
                depth1_pars,
            ),
            ConvUnit(
                depth1_OS,
                depth2_pars,
            ),
        )

    def get_img_combine_module(self):
        assert self.rgb_OS is not None, "RGB module not initiated yet"
        assert self.depth_OS is not None, "Depth module not initiated yet"
        assert self.rgb_OS == self.depth_OS, \
            "RGB module output shape " + self.rgb_OS  + " and" +\
            " depth module output shape " + self.depth_OS + " don't match"
        img_combine_IS = self.rgb_OS

        # Image Combination Module --------------------------------------------
        img_combine1_pars = ConvUnitPars(16, 3, 1, 2, 4, 2)
        img_combine2_pars = ConvUnitPars(16, 2, 1, 2, 4, 2)
        print("---------------\nCombine layers output shapes")
        img_combine1_OS = get_cu_OS(img_combine_IS, img_combine1_pars)
        img_combine2_OS = get_cu_OS(
            img_combine1_OS, img_combine2_pars)
        self.img_combine_OS = np.prod(img_combine2_OS)
        print(img_combine_IS, img_combine1_OS, img_combine2_OS,
              self.img_combine_OS)
        print("---------------")
        return nn.Sequential(
            ConvUnit(
                img_combine_IS,
                img_combine1_pars,
            ),
            ConvUnit(
                img_combine1_OS,
                img_combine2_pars,
            ),
            nn.Flatten(),
        )

    def get_joint_module(self):
        # Joint Module --------------------------------------------------------
        self.joint_OS = 64
        return nn.Sequential(
            nn.Linear(self.num_joints, 32),
            nn.Linear(32, self.joint_OS),
        )

    def get_pillow_module(self):
        # Pillow Module -------------------------------------------------------
        self.pillow_OS = 64
        return nn.Sequential(
            nn.Linear(self.pillow_pose_size, 32),
            nn.Linear(32, self.pillow_OS),
        )

    def get_pillow_joint_combine_module(self):
        assert self.pillow_OS is not None, "Pillow module not initiated yet"
        assert self.joint_OS is not None, "RGB module not initiated yet"
        assert self.img_combine_OS is not None, "Img Combine module not initiated yet"
        # Pillow + Joint Combination Module ---------------------------------
        return nn.Sequential(
            nn.Linear(self.pillow_OS + self.joint_OS, self.img_combine_OS)
        )

    def get_full_combine_module(self):
        # Full Combination Module ----------------------------------------------
        return nn.Sequential(
            nn.Linear(self.img_combine_OS, 64),
            nn.Linear(64, self.full_combine_OS),
        )


if __name__ == "__main__":
    # Demonstrating forward pass of ActorCritic -------------------------------
    from rl_base import get_sample_state

    # Initializing the Actor Critic
    img_wh = [512, 512]
    AC = ActorCritic(
        rgb_img_shape=(img_wh[0], img_wh[1], 3),
        depth_img_shape=(img_wh[0], img_wh[1], 1),
        num_joints=6,
        pillow_pose_size=3,
        num_outputs=6
    )

    # Run a forward pass
    state = get_sample_state(img_wh[0], img_wh[1])
    action_dist, value = AC(state.to_tensor())

    print(action_dist)
    print(value)
