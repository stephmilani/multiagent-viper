import random

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False, eval=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if eval:
            action = self.policy.eval()(obs)
        else:
            action = self.policy(obs)
        if self.discrete_action:
            if explore:
                if random.uniform(0, 1) < self.exploration:
                    action = torch.nn.functional.one_hot(
                        torch.randint(0, action.shape[1], size=(action.shape[0],), device=action.device),
                        num_classes=action.shape[1]).float()
                else:
                    action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class FittedQDTAgent(object):
    def __init__(self, training_data, max_depth=2, max_iters=10,
                 gamma=0.95, init_q_val=0.0, n_acts=5):
        self.max_iters = max_iters
        self.training_data = training_data
        self.q = {}
        self.init_q = init_q_val
        self.gamma = gamma
        self.n_acts = n_acts
        self.max_depth = max_depth
        self.clf = DecisionTreeRegressor(max_depth=max_depth)

    def step(self, obs, explore=False):
        if len(np.array(obs).shape) > 1:
            obs = np.array(obs)[0]
        else:
            obs = np.array(obs)
        bins = [-1., -.75, -.5, -.25, 0, .25, .5, .75, 1., np.inf]
        binned_obs = []
        for entry in obs:
            lb = -np.inf
            for b, bin in enumerate(bins):
                ub = bin
                if entry < ub and entry >= lb:
                    binned_obs.append(b)
                    break
                lb = ub
        assert len(binned_obs) == len(obs)
        obs = np.array(binned_obs)

        # we will only ever step during evaluation, so no need to introduce action noise
        obs = np.array(obs)
        if np.array2string(obs) in self.q.keys():
            all_acts = self.q[np.array2string(obs)]
        else:
            all_acts = np.zeros(self.n_acts)
        best_act = np.argmax(all_acts)
        arr_act = np.zeros(self.n_acts)
        arr_act[best_act] = 1.
        return torch.Tensor(arr_act)

    def train(self, train_data=None):
        # can pass in training data here if desired
        if train_data is not None:
            self.training_data = train_data

        # do some number of iterations of fitted q
        for it in range(self.max_iters):
            if it % 20 == 0:
                print("Iteration: " + str(it))
            x, y = [], []
            for l in range(len(self.training_data)):
                s, a, r, ns = self.training_data[l]
                s = np.array(s)
                i_l = np.append(s, a)
                if np.array2string(ns) not in self.q.keys():
                    self.q[np.array2string(ns)] = np.full(self.n_acts, self.init_q)
                    # TODO: one challenge of init this way is if q-vals tend to be very negative we may end up with an alg that chooses the actionms that the agent never takes in the dataset
                o_l = r + self.gamma + np.max(self.q[np.array2string(ns)])
                x.append(i_l)
                y.append(o_l)
                if it == 99 and (l == 4 or l == 10):
                    print(self.q[np.array2string(ns)])

            assert len(x) == len(y)

            self.clf.fit(x, y)
            for l in range(len(self.training_data)):
                s, a, r, ns = self.training_data[l]
                s = np.array(s)
                if np.array2string(s) not in self.q.keys():
                    self.q[np.array2string(s)] = np.full(self.n_acts, self.init_q)
                self.q[np.array2string(s)][np.argmax(a)] = self.clf.predict([np.append(s, a)])
        # TODO: incoprorate pruning step here


class ImitatedDTAgent(object):
    def __init__(self, training_data, max_depth=2, max_iters=10, n_acts=5):
        self.max_iters = max_iters
        self.training_data = training_data
        self.max_depth = max_depth
        self.n_acts = n_acts
        self.clf = DecisionTreeClassifier(max_depth=max_depth)

    def step(self, obs, explore=False):
        if not isinstance(obs, np.ndarray):
            obs = obs.cpu().detach().numpy().squeeze()
        act = self.clf.predict([obs])
        return act[0]

    def train(self, train_data=None):
        if train_data is not None:
            self.training_data = train_data

        states, actions = [], []
        for datapoint in self.training_data:
            s, a, r, ns = datapoint
            states.append(s)
            actions.append(a)

        assert len(actions[0]) == self.n_acts
        self.clf.fit(states, actions)
