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
from DQN_model import *

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

def train(Q, Q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(5)
        rgb, joint, pillow, goal, gripper = concat_obs(s)
        q_out = Q(rgb, joint, pillow, goal, gripper)
        a = torch.tensor(a)
        a = torch.reshape(a , (-1,1))
        # print("Q_out", q_out)
        q_a = q_out.gather(1, a)
        # print("q_a", q_a)
        rgb_p, joint_p, pillow_p, goal_p, gripper_p = concat_obs(s_prime)
        q_out_p = Q_target(rgb_p, joint_p, pillow_p, goal_p, gripper_p)
        target = torch.tensor(r) + gamma * q_out_p.max(1)[0] * torch.tensor(done_mask)
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

# if __name__ == '__main__':
def minimal_rl():
    print("Inside DQN_train.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('DQN_Train_Test_Node',
                    anonymous=True)
    env = Niryo()
    memory = ReplayBuffer()
    Q = DQN_model(3,6,14)
    Q_target = DQN_model(3,6,14)

    score = 0.0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    for n_epi in range(1):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = Q.sample_action(s,epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(Q,Q_target, memory, optimizer)



    # Testing Process
    # epsilon = max(0.01, 0.08 - 0.01*(1/200)) #Linear annealing from 8% to 1%
    # s = env.reset()
    # done = False
    # a = Q.sample_action(s,epsilon)
    # s_prime, r, done, info = env.step(a)
    # done_mask = 0.0 if done else 1.0
    # print(s_prime)
    # print(r)
    # print(done_mask)
    # print(info)
    # print("Done Testing")

if __name__ == '__main__':
    print("Inside DQN_train.py")
    rospy.loginfo("Initialize Niryo RL Node")
    rospy.init_node('DQN_Train_Test_Node',
                    anonymous=True)
    env = Niryo(randomspawn=True)
    memory = ReplayBuffer()
    Q = DQN_model(3,6,14)
    Q_target = DQN_model(3,6,14)
    print_interval = 1

    score = 0.0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    for n_epi in range(5):
        # epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        epsilon = 0.8
        s = env.reset()
        done = False
        # print(s.rgb.size(), s.pillow.size(), s.goal.size(), s.joint.size(), s.gripper.size())

        for i in range(10): #change this to while not done
            a = Q.sample_action(s,epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 7: #modify this number as desired
            train(Q,Q_target, memory, optimizer)

        if n_epi%print_interval == 0 and n_epi!=0:
            Q_target.load_state_dict(Q.state_dict())
            Q_target.save_model()
            print("n_episode: ", n_epi)
            print("Score: ", score)
            score = 0.0
            

    # train
    # s, a, r, s_prime, done_mask = memory.sample(5)
    # rgb, joint, pillow, goal, gripper = concat_obs(s)
    # q_out = Q(rgb, joint, pillow, goal, gripper)
    # a = torch.tensor(a)
    # a = torch.reshape(a , (-1,1))
    # print("Q_out", q_out)
    # q_a = q_out.gather(1, a)
    # print("q_a", q_a)
    # rgb_p, joint_p, pillow_p, goal_p, gripper_p = concat_obs(s_prime)
    # q_out_p = Q_target(rgb_p, joint_p, pillow_p, goal_p, gripper_p)
    # target = torch.tensor(r) + gamma * q_out_p.max(1)[0] * torch.tensor(done_mask)
    # target = target.reshape((-1,1))
    # print("Target:", target)
    # loss = nn.MSELoss()
    # output = loss(target, q_a)
    # print("loss", output)

    # optimizer.zero_grad()
    # output.backward()
    # optimizer.step()
    # print("done all process")


    # print("Printing Obs Concat")
    # print("RGB", rgb.size())
    # print("Pillow", pillow.size())
    # print("Goal", goal.size())
    # print("Joint", joint.size())
    # print("Gripper", gripper.size())


    # # print(memory.buffer[0])
    # # print(memory.buffer[0].rgb.shape)
    # epsilon = 0.8
    # s = env.reset()
    # done = False
    # a = Q.sample_action(s,epsilon)
    # s_prime, r, done, info = env.step(a)
    # print("Printing State: ")
    # print("RGB", s_prime.rgb)
    # print("Pillow", s_prime.pillow)
    # print("Goal", s_prime.goal)
    # print("Joint", s_prime.joint)
    # print("Gripper", s_prime.gripper)
    # print(s_prime.rgb.size())
    # print("Prinitng Memory Sample: ")
    # print(memory.sample(5))
    # print("Printing S:")
    # print(s)
    # train(Q,Q_target, memory, optimizer)
    # s, a, r, s_prime, done = memory.sample(5)
    # print(s,a,r,s_prime,done)
    print("Done")