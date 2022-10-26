import numpy as np
import torch
from torch.autograd import Variable

class RandomAgent():
    def __init__(self, env, index, num_agents=3):
        self.num_agents = num_agents
        self.env = env
        self.index = index
        
    def step(self, state, explore=False):
        act = np.zeros(self.env.action_space[self.index].n)
        pred = self.env.action_space[self.index].sample()
        act[pred] = 1.
        return [torch.from_numpy(act)]
        #return self.env.action_space[self.index].sample()
    def predict(self, state, explore=False):
        act = np.zeros(self.env.action_space[self.index].n)
        pred = self.env.action_space[self.index].sample()
        print(pred)
        act[pred] = 1.
        return [torch.from_numpy(act)]
    
