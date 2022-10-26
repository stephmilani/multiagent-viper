import itertools
import time
from copy import copy

import numpy as np
import torch
from util.envs import get_input_shape, get_actions


class TeacherFunc:
    def __init__(self, env, decision_func, index, default_q=-1e10, chosen_q=-1e-10, team=None):
        self.env = env
        self.input_shape = get_input_shape(env, index)
        self.num_actions = get_actions(env, index)
        self.policy = decision_func
        self.decision_func = decision_func.critic.eval()  # NOTE: updated the decision func to be the critic not the policy
        self.index = index
        self.default_q = default_q
        self.chosen_q = chosen_q
        self.team = team

    def predict_q(self, inputs):
        # print('inp shape', inputs)
        # try:
        #    assert np.asarray(inputs).shape[1:] == self.input_shape
        # except AssertionError:
        #    assert np.asarray(inputs).shape == self.input_shape
        num_inp = np.asarray(inputs).shape[0]
        qs = np.ones((num_inp, self.num_actions)) * self.default_q
        predictions = self.decision_func(inputs)
        # get the predictions for each state input 
        # predictions = []
        # print('inputs are', inputs.shape)
        # for i, inp in enumerate(inputs):
        # predict = np.asarray(self.get_action_probs(inp))
        #    predict = np.asarray(self.decision_func(inp))
        #     predictions.append(predict)
        #    qs[i, np.arange(self.num_actions)] = np.log(predict + 1e-10)
        # try:
        #    assert np.asarray(qs).shape == (num_inp, self.num_actions)
        # except AssertionError:
        #     pass # update this 
        # return qs
        # print('qs are', predictions.data.numpy().shape)
        return predictions.data.numpy()

    def predict_q_for_each_action(self, obss, acts):
        acts = copy(acts)
        predictions = []
        for cur_action in range(self.num_actions):
            acts[self.index] = torch.eye(self.num_actions)[cur_action].repeat((acts[self.index].shape[0], 1))
            inputs = torch.cat((*obss, *acts), dim=1)
            predictions.append(self.decision_func(inputs))
        return torch.cat(predictions, dim=1).detach().numpy()

    def predict_optimal_others(self, obss, acts):
        # Assuming acts is generated by teacher
        obss, acts = [obs.detach().clone() for obs in obss], [act.detach().clone() for act in acts]
        device = obss[0].device
        self.decision_func = self.decision_func.to(device)
        ret1, ret2 = [], []
        start = time.time()

        cur_team = [k for k, v in self.team.items() if v == self.team[self.index]]
        others = [i for i in range(len(self.team)) if i not in cur_team]

        for i in range(obss[0].shape[0]):
            cur_obss = [obss[agent][i] for agent in range(len(obss))]
            cur_acts = [acts[agent][i] for agent in range(len(acts))]
            action_size = [len(act) for act in cur_acts]
            # cur_acts_backup = copy(cur_acts)

            # Minimization over agents outside current team
            self._generate_mask([action_size[_] for _ in others])
            ret1.append(self._generate_other_actions(cur_obss, cur_acts, cur_team, others))

            worst = []
            for team_actions in itertools.product(*[list(range(action_size[_])) for _ in cur_team]):
                for j in range(len(team_actions)):
                    cur_acts[cur_team[j]] = torch.eye(action_size[cur_team[j]])[team_actions[j]].to(device)
                average = self._generate_other_actions(cur_obss, cur_acts, cur_team, others)
                worst.append(average)
            ret2.append(torch.stack(worst))
        # print('Assigning time:', time.time() - start)

        r1 = torch.min(torch.squeeze(self.decision_func(torch.stack(ret1)), dim=-1), dim=1)[0]
        r2 = torch.min(torch.mean(torch.squeeze(self.decision_func(torch.stack(ret2)), dim=-1), dim=-1), dim=-1)[0]
        ret = torch.Tensor(r1 - r2).view(-1).clone().detach().numpy()
        r1 = r2 = ret1 = ret2 = None
        return ret

    def _generate_mask(self, action_sizes):
        # Generate one-hot mask and cache it
        if hasattr(self, 'other_mask') and self.other_mask is not None and action_sizes == self.other_mask[1]:
            return
        if len(action_sizes) == 0:
            self.other_mask = None
            return
        self.other_mask = [list() for _ in range(len(action_sizes))]
        for other_actions in itertools.product(*[list(range(size)) for size in action_sizes]):
            for j in range(len(other_actions)):
                self.other_mask[j].append(torch.eye(action_sizes[j])[other_actions[j]])
        for j in range(len(self.other_mask)):
            self.other_mask[j] = torch.stack(self.other_mask[j])
        self.other_mask = (self.other_mask, action_sizes)

    def _generate_other_actions(self, obss, acts, cur_team, others):
        if self.other_mask is None:
            return torch.cat((*obss, *acts)).reshape(1, -1)
        other_actions = self.other_mask[0]
        repeat_times = len(other_actions[0])
        average_obs = [obs.repeat(repeat_times, 1) for obs in obss]
        average_acts = [act.repeat(repeat_times, 1) if i in cur_team else other_actions[others.index(i)] for i, act in
                        enumerate(acts)]
        return torch.cat((*average_obs, *average_acts), dim=1).clone().detach()

    def predict_average_others(self, obss, acts):
        # Assuming acts is generated by teacher
        obss, acts = [obs.detach().clone() for obs in obss], [act.detach().clone() for act in acts]
        device = obss[0].device
        self.decision_func = self.decision_func.to(device)
        ret1, ret2 = [], []

        cur_team = [k for k, v in self.team.items() if v == self.team[self.index]]
        others = [i for i in range(len(self.team)) if i not in cur_team]

        for i in range(obss[0].shape[0]):
            cur_obss = [obss[agent][i] for agent in range(len(obss))]
            cur_acts = [acts[agent][i] for agent in range(len(acts))]
            action_size = [len(act) for act in cur_acts]

            # Minimization over agents outside current team
            self._generate_mask([action_size[_] for _ in others])
            ret1.append(self._generate_other_actions(cur_obss, cur_acts, cur_team, others))

            worst = []
            for team_actions in itertools.product(*[list(range(action_size[_])) for _ in cur_team]):
                for j in range(len(team_actions)):
                    cur_acts[cur_team[j]] = torch.eye(action_size[cur_team[j]])[team_actions[j]].to(device)
                average = self._generate_other_actions(cur_obss, cur_acts, cur_team, others)
                worst.append(average)
            ret2.append(torch.stack(worst))

        r1 = torch.mean(torch.squeeze(self.decision_func(torch.stack(ret1)), dim=-1), dim=1)
        r2 = torch.min(torch.mean(torch.squeeze(self.decision_func(torch.stack(ret2)), dim=-1), dim=-1), dim=-1)[0]
        ret = torch.Tensor(r1 - r2).view(-1).clone().detach().numpy()
        r1 = r2 = ret1 = ret2 = None
        return ret

    def iviper_resampling(self, obss, acts):
        r2 = []
        for i in range(len(acts[self.index][0])):
            acts[self.index] = torch.eye(self.num_actions)[i].repeat((acts[self.index].shape[0], 1))
            inputs = torch.cat((*obss, *acts), dim=1)
            r2.append(self.decision_func(inputs))
        ret = torch.max(torch.concat(r2, axis=1), dim=1)[0].reshape(-1, 1) - torch.min(torch.concat(r2, axis=1), dim=1)[0].reshape(-1, 1)
        return ret.reshape(-1).clone().detach().numpy()

    def predict_max_min_minus_min_max(self, obss, acts):
        obss, acts = copy(obss), copy(acts)
        ret = []
        for i in range(obss[0].shape[0]):
            cur_obss = [obss[agent][i] for agent in range(len(obss))]
            cur_acts = [acts[agent][i] for agent in range(len(acts))]
            all_best, all_worst = [], []
            for ego_action in range(self.num_actions):
                best, worst = -1e5, 1e5
                for other_actions in itertools.product(*[list(range(self.num_actions)) for _ in range(len(acts) - 1)]):
                    cur_acts[self.index] = torch.eye(self.num_actions)[ego_action]
                    cnt = -1
                    for j in range(len(acts)):
                        if j == self.index:
                            continue
                        cnt += 1
                        cur_acts[j] = torch.eye(self.num_actions)[other_actions[cnt]]
                    q = self.decision_func(torch.cat((*cur_obss, *cur_acts))).data.numpy()[0]
                    best, worst = max(best, q), min(worst, q)
                all_best.append(best)
                all_worst.append(worst)
            ret.append(max(all_worst) - min(all_best))
        return torch.Tensor(ret).view(-1).numpy()

    def predict(self, inputs):
        predictions = self.get_action_probs(inputs)
        return predictions
        # print('predictions are', predictions)
        # return np.argmax(predictions, axis=1)

    # action dist
    # TODO:
    # change this to output action probs
    # rename to "get action probs"
    # set the q func above to use this function (under new name)
    # make a "predict" func that calls "get action probs" and performs argmax along app axis
    def get_action_probs(self, inputs):
        # inputs = np.asarray(inputs)
        # print('shape of is', np.array(inputs).shape)
        # print('input shape is supposed to be', self.input_shape)
        try:
            assert np.asarray(inputs).shape[1:] == self.input_shape
        except AssertionError:
            # assert np.asarray(inputs).shape[1:] == self.input_shape
            assert np.asarray(inputs).shape == self.input_shape
        # print("input shape in predict is", self.input_shape)
        if len(inputs.shape) > 1:
            num_inp = np.asarray(inputs).shape[0]
        else:
            num_inp = 1

        inputs = inputs.reshape(num_inp, *self.input_shape)
        outputs = []
        # print('inputs are', inputs.shape)
        # for i in range(num_inp):
        #    inp = inputs[i]
        # self.decision_func.policy.eval()
        # inputs = Variable(torch.Tensor(inputs).view(1, -1), requires_grad=False)
        acts = self.policy.step(inputs, explore=False, eval=True)
        outputs.append(acts)
        return acts

    def train(self, num_timesteps=1000):
        self.underlying_teacher.learn(total_timesteps=num_timesteps)