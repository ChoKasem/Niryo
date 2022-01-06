import collections
import random
import torch

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_p_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_p, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_p_lst.append(s_p)
            done_mask_lst.append(done_mask)

        return s_lst, torch.tensor(a_lst),\
               torch.tensor(r_lst), s_p_lst, \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)