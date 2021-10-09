import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import rospy

import sys
sys.path.append("..")

from envs import Arm, World, Gripper

class DQN_model(nn.Module):
    def __init__(self, img_ch, num_joint, num_action):
        super(DQN_model, self).__init__()
        self.img_net = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(0,-1),
        )

        self.pillow_net = nn.Sequential(
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )

        self.goal_net = nn.Sequential(
            nn.Linear(7, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        
        self.joint_net = nn.Sequential(
            nn.Linear(num_joint, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        self.gripper_net = nn.Sequential(
            nn.Linear(1,5),
            nn.ReLU(),
            nn.Linear(5,3),
            nn.ReLU(),
        )

        self.action_net = nn.Sequential(
            nn.Linear(2034, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_action)
        )

    def forward(self, img, joint, pillow, goal, gripper):
        img = torch.reshape(img, (-1,3,480,640))
        img_out = self.img_net(img)
        # img_out = torch.flatten(img_out)
        pillow_out = self.pillow_net(pillow)
        goal_out = self.goal_net(goal)
        joint_out = self.joint_net(joint)
        gripper_out = self.gripper_net(gripper)

        # Concatenate in dim1 (feature dimension)
        # print(img_out.shape)
        # print(pillow_out)
        # print(pillow_out.shape)
        # print(goal_out.shape)
        # print(joint_out.shape)
        # print(gripper_out.shape)
        
        out = torch.cat((img_out, pillow_out, goal_out, joint_out, gripper_out))
        print(out.shape)
        out = self.action_net(out)
        # print(out.shape)
        return out

def pose2vector(pose):
    vec = [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return vec


if __name__ == '__main__':
    print("Inside DQN_model.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('DQN_Model_Test_Node',
                    anonymous=True)
    arm = Arm()
    world = World()
    Gripper = Gripper()
    # # qnet = DQN_model()
    # print("Arm Ready")

    # image
    # print(arm.image.shape)
    # # print(arm.image)
    img = torch.from_numpy(arm.image).float()
    # print(img_torch)

    # joint
    # print("Print Joint")
    # print(arm.joint_angle)
    joint = arm.joint_angle.position[:6]
    joint = torch.tensor(joint)
    # print(joint_torch)
    # print(joint_torch.type())

    # pillow pose
    pillow_pose = pose2vector(world.pillow_pose.pose)
    pillow_pose = torch.tensor(pillow_pose)
    
    # goal pose
    goal_pose = pose2vector(world.goal_pose.pose)
    goal_pose = torch.tensor(goal_pose)

    # gripper
    gripper = torch.tensor([Gripper.gripper_angle]).float()


# mock data
    # img = torch.randn(480, 640, 3)
    # x = torch.reshape(img, (-1,3,480,640))
    # print(x.shape)
    # joint = torch.randn(6)
    # pillow_pose = torch.randn(7)
    # goal_pose = torch.randn(7)
    # gripper = torch.randn(1)

    model = DQN_model(3,6,14)
    # print(model)
    output = model(img, joint, pillow_pose, goal_pose, gripper)
    print("Model Output")
    print(output)
    print("Done")