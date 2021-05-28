"""
Adapted from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
"""

from rl_base import State
import torch


class PPO:
    def __init__(self, agent, agent_old, n_epochs=10, mini_batch_size=32, buffer_size=30):
        self.data = []
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.eps_clip = 0.2
        self.agent = agent
        self.gamma = 0.9
        self.lmbda = 0.9

        self.MseLoss = torch.nn.MSELoss()


        lr = 0.0003 
        betas = (0.9, 0.999)
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr, betas=betas)
        
        self.agent_old = agent_old
        self.agent_old.load_state_dict(self.agent.state_dict())

    def update(self, memory):
        def list_to_tensor(tensor_list):
            return torch.squeeze(torch.stack(tensor_list), 1)

        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensors
        old_states = State.to_tensor_object(memory.states).transform(list_to_tensor)
        old_actions = list_to_tensor(memory.actions)
        old_logprobs = list_to_tensor(memory.logprobs)
        
        # Optimize policy for n epochs:
        for _ in range(self.n_epochs):
            # Evaluating old actions and values :
            action_dist, state_values = self.agent(old_states)
            state_values = torch.squeeze(state_values)
            logprobs = action_dist.log_prob(old_actions)
            dist_entropy = action_dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.agent_old.load_state_dict(self.agent.state_dict())