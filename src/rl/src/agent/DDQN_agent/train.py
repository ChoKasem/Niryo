# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import numpy as np
# import rospy
# import datetime

import torch.optim as optim
import sys
sys.path.append("../..")

from envs import Arm, World, Gripper, Niryo, ActionSpace, ObservationSpace
from utils.ReplayBuffer import ReplayBuffer
from DDQN_model import *

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

def train(Q, Q_target, memory, optimizer):
    for i in range(20):
        s, a, r, s_prime, done_mask = memory.sample(20)
        rgb, joint, pillow, goal, gripper = concat_obs(s)
        q_out = Q(rgb, joint, pillow, goal, gripper)
        a = torch.tensor(a)
        a = torch.reshape(a , (-1,1))
        q_a = q_out.gather(1, a)
        rgb_p, joint_p, pillow_p, goal_p, gripper_p = concat_obs(s_prime)

        # choose action base on online network of next state
        q_out_p = Q(rgb_p, joint_p, pillow_p, goal_p, gripper_p)
        a_q_out_max = q_out_p.max(1)[1]

        # evaluate the action base on target network
        q_target_out_p_prob = Q_target(rgb_p, joint_p, pillow_p, goal_p, gripper_p)
        q_target_out_p = q_target_out_p_prob.gather(1,a_q_out_max.reshape(-1,1))
        q_target_out_p = q_target_out_p.flatten()
        target = torch.tensor(r) + gamma * q_target_out_p * torch.tensor(done_mask)
        target = target.reshape((-1,1))
        # print("Target:", target)
        loss = nn.MSELoss()
        output = loss(target, q_a)
        # print("loss", output)

        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        print("done training iteration: ", i)

def concat_obs(s_lst):
    rgb, joint, pillow, goal, gripper = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    for s in s_lst:
        rgb = torch.cat((rgb,s.rgb))
        joint = torch.cat((joint,s.joint))
        pillow = torch.cat((pillow, s.pillow))
        goal = torch.cat((goal,s.goal))
        gripper = torch.cat((gripper, s.gripper))

        # rgb.append(s.rgb)
        # joint.append(s.joint)
        # pillow.append(s.pillow)
        # goal.append(s.goal)
        # gripper.append(s.gripper)

    rgb = torch.reshape(rgb, (-1, 3, 480, 640))
    joint = torch.reshape(joint, (-1, 6))
    pillow = torch.reshape(pillow, (-1, 7))
    goal = torch.reshape(goal, (-1,7))
    

    # print("RGB:", rgb.size(), rgb)
    # print("Joint:", joint.size(), joint)
    # print("Pillow:", pillow.size(), pillow)
    # print("Goal:", goal.size(), goal)
    # print("Gripper:", gripper.size(), gripper)


    return rgb, joint, \
           torch.tensor(pillow), torch.tensor(goal), torch.tensor(gripper)

if __name__ == '__main__':
    print("Inside DDQN_train.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('DDQN_Train_Test_Node',
                    anonymous=True)
    env = Niryo(randomspawn=True)
    memory = ReplayBuffer()
    Q = DDQN_model(3,6,14)
    Q_target = DDQN_model(3,6,14)
    print_interval = 1

    score = 0.0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    for n_epi in range(50):
        # epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        epsilon = 0.8
        s = env.reset()
        done = False
        # print(s.rgb.size(), s.pillow.size(), s.goal.size(), s.joint.size(), s.gripper.size())

        while not done: #change this to while not done
            a = Q.sample_action(s,epsilon)
            print("Doing action: ", a)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 50: #modify this number as desired
            train(Q,Q_target, memory, optimizer)

        if n_epi%print_interval == 0 and n_epi!=0:
            Q_target.load_state_dict(Q.state_dict())
            Q_target.save_model()
            print("n_episode: ", n_epi)
            print("Score: ", score)
            score = 0.0
            

   
    print("Done")