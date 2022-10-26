from typing import Dict

import gym
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from gym.spaces import Tuple, Box, Discrete
from copy import deepcopy
import torch
from torch.autograd import Variable
from multiagent.environment import MultiAgentEnv
import numpy as np
from core.dt import *
import time

from maviper.python.viper.core.joint_dt import JointDTPolicy


class SingleAgentWrapper():
    def __init__(self, multi_agent_env, agents, index, full_state_info, env_name):
        self.env = multi_agent_env
        self.agents = agents
        self.agent_index = index
        self.env_shape = np.asarray(self.env.observation_space).shape
        self.step_sequence = []
        self.current_obs = None
        self.all_prev_acts = None
        self.all_agent_rews = None
        self.agent_rews = []  # [0 for _ in range(2)] #[]
        self.full_state_info = full_state_info
        self.current_obs = self.reset()
        self.env_name = env_name  # TODO: HARDCODED

    def get_obs_shape(self):
        if self.full_state_info:
            return np.array(self.env.observation_space).shape
        else:
            return self.env.observation_space[self.agent_index].shape

    def get_action_shape(self):
        return self.env.action_space[self.agent_index].n

    def step(self, action):
        actions = []
        # for each agent 
        for i, agent in enumerate(self.agents):
            if i != self.agent_index:
                if isinstance(agent, DTPolicy):
                    ag_act = agent.predict(np.array([self.current_obs[i]]))
                    actions.append(ag_act[0])
                else:
                    actions.append(agent.step(
                        Variable(torch.Tensor(self.current_obs[i]).view(1, -1),
                                 requires_grad=False)).data.numpy().flatten()
                                   )
            else:
                if isinstance(action, np.ndarray):
                    actions.append(action)
                else:
                    actions.append(action.data.numpy().flatten())
        assert len(actions) == len(self.agents)
        if self.env_name == 'simple_cybersec':
            actions = [actions]
        all_obs, all_rew, all_dones, all_info = self.env.step(actions)
        if self.env_name == 'simple_cybersec':
            all_obs = all_obs[0]
            all_rew = all_rew[0]
            all_dones = all_dones[0]
            actions = actions[0]
        self.current_obs = all_obs
        # if self.env_name == 'simple_cybersec':
        #    self.current_obs = all_obs[0]
        self.all_prev_acts = actions
        self.all_agent_rews = all_rew
        # if self.env_name == 'simple_cybersec':
        #    all_rew = all_rew[0]
        self.agent_rews += all_rew
        self.step_sequence.append([all_obs, all_rew, all_dones, all_info])
        if self.full_state_info:
            return (all_obs, all_rew, all_dones, all_info)
        else:
            if self.env_name == 'simple_cybersec':
                return (all_obs[self.agent_index], all_rew[self.agent_index],
                        all_dones, all_info)
            else:
                return (all_obs[self.agent_index], all_rew[self.agent_index],
                        all_dones[self.agent_index], all_info)

    def reset(self):
        self.current_obs = self.env.reset()
        self.agent_rews, self.step_sequence, self.all_prev_acts = [], [], []
        # self.agent_rews = [0 for _ in range(2)] # TODO added for csec
        if self.full_state_info:
            return self.current_obs
        else:
            return self.current_obs[self.agent_index]

    def seed(self, seed):
        self.env.seed(seed)


class ExploitabilityWrapper:
    def __init__(self, env, existing_agents):
        self.env = env

        if all([hasattr(a, 'adversary') for a in env.agents]):
            all_agent_types = ['adversary' if a.adversary else 'agent' for a in env.agents]
        else:
            all_agent_types = ['agent' for _ in env.agents]

        self.total_agents = len(self.env.agents)
        self.missing_agent = [agent for agent in range(self.total_agents) if agent not in existing_agents]
        self.agent_types = [all_agent_types[i] for i in self.missing_agent]
        self.existing_agent = existing_agents
        self.action_space = []
        self.observation_space = []
        for agent in self.missing_agent:
            self.action_space.append(self.env.action_space[agent])
            self.observation_space.append(self.env.observation_space[agent])

    def step(self, action, obs, device='cpu'):
        all_action = [None] * self.total_agents
        for i in range(len(action)):
            all_action[self.missing_agent[i]] = action[i]
        for i, agent in self.existing_agent.items():
            if hasattr(agent, 'predict'):
                obs[i] = obs[i].to('cpu')
                action = agent.predict(obs[i])
            elif hasattr(agent, 'step'):
                try:
                    agent.policy.to(device)
                except:
                    pass
                action = agent.step(obs[i]).reshape(1, -1)
            else:
                raise NotImplementedError()
            all_action[i] = action
        all_action = [ac[0] for ac in all_action]
        return self.env.step(all_action)

    def get_missing_agent(self):
        return self.missing_agent

    def __getattr__(self, item):
        return getattr(self.env, item)
