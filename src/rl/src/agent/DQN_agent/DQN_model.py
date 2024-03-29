import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import rospy
import datetime
import random

import sys
sys.path.append("../..")

from envs import Arm, World, Gripper, Niryo

class DQN_model(nn.Module):
    def __init__(self, img_ch, num_joint, num_action):
        super(DQN_model, self).__init__()
        self.num_acition = num_action
        now = datetime.datetime.now()
        self.savefile = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute) + '.pt'
        self.img_net = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 1, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
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
            nn.Linear(81, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_action)
        )

    def obs_forward(self, obs):
        return self.forward(obs.rgb, obs.joint, obs.pillow, obs.goal, obs.gripper)

    def forward(self, img, joint, pillow, goal, gripper):
        # img = torch.reshape(img, (-1,3,480,640))
        img_out = self.img_net(img)
        # print("Img:", img_out.size(), img_out)
        pillow_out = self.pillow_net(pillow)
        # print("Pillow:", pillow_out.size(), pillow_out)
        goal_out = self.goal_net(goal)
        # print("Goal:", goal_out.size(), goal_out)
        joint_out = self.joint_net(joint)
        # print("Joint:",joint_out.size(), joint_out)
        gripper_out = self.gripper_net(gripper)
        # print("Gripper:",gripper_out.size(), gripper_out)

        # Concatenate in dim1 (feature dimension)
        # print(img_out.shape)
        # print(pillow_out)
        # print(pillow_out.shape)
        # print(goal_out.shape)
        # print(joint_out.shape)
        # print(gripper_out.shape)
        
        out = torch.cat((img_out, pillow_out, goal_out, joint_out, gripper_out),1)
        out = self.action_net(out)
        # print(out.shape)
        return out

    def sample_action(self, obs, epsilon):
        out = self.obs_forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,self.num_acition-1)
        else:
            return out.argmax().item()

    def save_model(self):
        torch.save(self.state_dict(), 'save_model/' + self.savefile)

    def load_model(self, PATH):
        self.load_state_dict(torch.load(PATH))

def pose2vector(pose):
    vec = [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return vec


if __name__ == '__main__':
    print("Inside DQN_model.py")
    # rospy.loginfo("Initialize Niryo RL Node")
    # rospy.init_node('DQN_Model_Test_Node',
                    # anonymous=True)
    # arm = Arm()
    # world = World()
    # Gripper = Gripper()
    # env = Niryo()
    # print("done making env")
    # obs = env.get_obs()
    # # qnet = DQN_model()
    # print("Arm Ready")

    # image
    # print(arm.image.shape)
    # # print(arm.image)
    # img = torch.from_numpy(arm.image).float()
    # # print(img_torch)

    # # joint
    # # print("Print Joint")
    # # print(arm.joint_angle)
    # joint = arm.joint_angle.position[:6]
    # joint = torch.tensor(joint)
    # # print(joint_torch)
    # # print(joint_torch.type())

    # # pillow pose
    # pillow_pose = pose2vector(world.pillow_pose.pose)
    # pillow_pose = torch.tensor(pillow_pose)
    
    # # goal pose
    # goal_pose = pose2vector(world.goal_pose.pose)
    # goal_pose = torch.tensor(goal_pose)

    # # gripper
    # gripper = torch.tensor([Gripper.gripper_angle]).float()


# mock data
    # img = torch.randn(480, 640, 3)
    # x = torch.reshape(img, (-1,3,480,640))
    img = torch.randint(0,255, (1,3,480,640), dtype=torch.float32)
    # print(x.shape)
    joint = torch.randn((1,6))
    pillow_pose = torch.randn((1,7))
    goal_pose = torch.randn((1,7))
    gripper = torch.randn((1,1))

    model = DQN_model(3,6,14)
    output = model(img, joint, pillow_pose, goal_pose, gripper)
    print("Model Output")
    print(output)
    print("Done")

    # model.save_model()
    # print('save done')
    # model.load_model()
    # print('load done')