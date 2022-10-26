# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import time
from math import e

import numpy as np
from gym.spaces import Box
from util.log import *
import torch
from torch.autograd import Variable
from torch import Tensor
from core.dt import *
import os
from tqdm import tqdm
from algorithms.maddpg import MADDPG
from maviper.python.viper.core.dt import split_train_test
from maviper.python.viper.core.joint_dt import JointDTPolicy
from maviper.python.viper.core.teacher import TeacherFunc
from maviper.python.viper.util.env_wrappers import ExploitabilityWrapper, SingleAgentWrapper
from utils.agents import DDPGAgent
from utils.buffer import ReplayBuffer
import pandas as pd

SEED_NO = 0


def get_rollout(env, policy, render, filter_bad_episodes=False, scenario_name='simple_adversary', seed=0):
    # env.seed(SEED_NO)
    env.seed(seed)
    obs, done = np.array(env.reset()), False

    rollout = []
    max_timesteps = 25
    n_timesteps = 0
    successful_episode_agent = True
    epsilon = 0.1
    prev_acts = None
    info = None

    while not done:
        # Render
        if render:
            env.unwrapped.render()
        if isinstance(policy, TransformerPolicy):
            if hasattr(env, 'easy_feature_sizes'):
                info = [env.info_callback(agent, env.world)[1] for agent in env.agents]
            act = policy.predict(np.array([obs]), info)[0]
        else:
            if isinstance(obs[0], np.ndarray):
                torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False) for i in range(len(obs))]
            else:
                torch_obs = Variable(torch.Tensor(obs).view(1, -1), requires_grad=False)
            # Action
            if isinstance(policy, MADDPG):
                act = [ac.data.numpy().flatten() for ac in policy.predict(torch_obs)]
            else:
                if hasattr(env, 'easy_feature_sizes'):
                    # switch to using easy features
                    info = [env.info_callback(agent, env.world)[1] for agent in env.agents]
                    if isinstance(obs[0], np.ndarray):
                        torch_info = [Variable(torch.Tensor(info[i]).view(1, -1), requires_grad=False) for i in
                                      range(len(obs))]
                    else:
                        torch_info = Variable(torch.Tensor(info[0]).view(1, -1), requires_grad=False)
                else:
                    torch_info = torch_obs

                if isinstance(policy, JointDTPolicy.Individual):
                    act = policy.predict(torch_info)[0].flatten()
                elif isinstance(policy, list) and isinstance(policy[0], JointDTPolicy.Individual):
                    act = [p.predict(torch_info[i])[0].flatten() for i, p in enumerate(policy)]
                else:
                    if hasattr(policy, 'predict'):
                        act = policy.predict(torch_info)[0].flatten()
                    elif hasattr(policy, 'step'):
                        act = policy.step(torch_obs, torch_info)
                    else:
                        raise NotImplementedError
        # Step
        next_obs, rew, done_poss, info = env.step(act)
        # Rollout (s, x, a, A, r, info)

        if isinstance(act, list):  # dealing with joint action
            act = np.concatenate(act)
        if not isinstance(act, np.ndarray):
            act = act.data.numpy().flatten()
        else:
            act = act.flatten()
        # if (info['n'][1][0] < epsilon and info['n'][2][1] < epsilon) \
        #    or (info['n'][1][1] < epsilon and info['n'][2][0] < epsilon):
        #        successful_episode_agent = True
        try:
            prev_acts = env.all_prev_acts
        except:
            prev_acts = prev_acts if prev_acts is not None else np.zeros_like(act)
        rollout.append(
            (obs, env.current_obs, act, prev_acts, rew, info)
        )

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)
        prev_acts = act
        n_timesteps += 1
        # if our environment, we don't use timesteps as termination 
        if 'cybersec' in scenario_name:
            done = bool(sum([int(d) for d in done_poss]))
        else:
            if n_timesteps >= max_timesteps:
                break

    return rollout


def get_rollouts(env, policy, render, n_batch_rollouts, env_name, global_seed):
    rollouts = []
    # n_in_batch = 0
    # while n_in_batch <= n_batch_rollouts:
    #    added, rollout = get_rollout(env, policy, render)
    #    if added: 
    #        rollouts.extend(rollout)
    #        n_in_batch +=1
    for i in range(n_batch_rollouts):
        seed = i * global_seed
        rollouts.extend(get_rollout(env, policy, render, False, env_name, seed=seed))
    return rollouts


def get_joint_rollout(env, policies, render, env_name, seed=0):
    env._seed(seed)
    obs, done = env.reset(), False
    assert len(policies) == len(obs)
    rollout = []
    max_timesteps, n_timesteps = 25, 0

    while not done:
        acts = []
        for p_i, policy in enumerate(policies):
            if isinstance(policy, TransformerPolicy):
                act = policy.predict(np.array([obs[p_i]]))[0]
            else:
                torch_obs = Variable(torch.Tensor(obs[p_i]).view(1, -1), requires_grad=False)
                act = policy.predict(torch_obs)[0].data.flatten()
            acts.append(act)
        assert len(acts) == len(policies)

        next_obs, rew, _, info = env.step(acts)
        acts = [
            a.data.numpy().flatten() if not isinstance(a, np.ndarray) else a.flatten() for a in acts
        ]

        rollout.append(
            (obs, obs, acts, acts, rew, info)
        )
        obs = [np.array(n_ob) for n_ob in next_obs]
        n_timesteps += 1
        if n_timesteps >= max_timesteps:
            break
    return rollout


def get_full_ma_rollout(env, policies, render):
    obs, done = env.reset(), False
    rollout = []
    max_timesteps, n_timesteps = 25, 0
    while not done:
        acts = []
        for p_i, policy in enumerate(policies):
            if isinstance(policy, TransformerPolicy):
                act = policy.predict(np.array([obs[p_i]]))[0]
            else:
                torch_obs = Variable(torch.Tensor(obs[p_i]).view(1, -1), requires_grad=False)
                act = policy.predict(torch_obs)[0].data.flatten()
            acts.append(act)

        # Step
        next_obs, rew, _, info = env.step(acts)
        updated_acts = []
        for act in acts:
            if not isinstance(act, np.ndarray):
                act = act.data.numpy().flatten()
            else:
                act = act.flatten()
            updated_acts.append(act)
        rollout.append(
            (obs, updated_acts, rew, info)
        )

        # Update (and remove LazyFrames)
        obs = next_obs
        n_timesteps += 1
        if n_timesteps >= max_timesteps:
            break
    return rollout


def get_ma_rollout(env, policies, render):
    obs, done = env.reset(), False
    rollout = []
    max_timesteps = 25
    n_timesteps = 0
    while not done:
        # Render
        if render:
            env.unwrapped.render()
        acts = []
        for p_i, policy in policies:
            if isinstance(policy, TransformerPolicy):
                act = policy.predict(np.array([obs][p_i]))[0]
            else:
                torch_obs = Variable(torch.Tensor(obs[p_i]).view(1, -1), requires_grad=False)
                act = policy.predict(torch_obs)[0].data.flatten()
            acts.append(act)

        # Step
        next_obs, rew, done, info = env.step(acts)
        # Rollout (s, x, a, A, r, info)

        acts = [
            a.data.numpy().flatten() if not isinstance(a, np.ndarray) else a.flatten() for a in acts
        ]

        rollout.append(
            (obs, obs, acts, acts, rew, info)
        )

        # Update (and remove LazyFrames)
        obs = next_obs
        n_timesteps += 1
        if n_timesteps >= max_timesteps:
            break
    return rollout


def get_ma_rollouts(env, policies, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_ma_rollout(env, policies, render))
    return rollouts


def get_full_ma_rollouts(env, policies, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_full_ma_rollout(env, policies, render))
    return rollouts


def _ma_sample(obss, acts, qs, max_pts, is_reweight):
    # step 1: compute probs 
    qs = np.array([q.flatten() for q in qs])
    ps = (qs - np.min(qs, axis=1)) / (np.max(qs, axis=1) - np.min(qs, axis=1))

    # step 2: aggregate ps 
    aggregation_method = 'entropy'
    if aggregation_method == 'entropy':
        ps = np.mean(ps, axis=0)  # ps for each q make sure that it is the right shape
        ps = ps / np.sum(ps)
        assert len(ps) == len(obss)

    # step 3: sample points 
    if is_reweight:
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        idx = np.random.choice(
            len(obss), size=min(max_pts, np.sum(ps > 0), replace=False)
        )
    # step 3: obtain sampled indices
    return obss[idx], acts[idx], qs[idx]


def _sample_ma(obss, acts, qs, agent_idx, max_pts, is_reweight):
    # step 1: compute probs 
    qs = qs[agent_idx].flatten()
    ps = (qs - np.min(qs)) / (np.max(qs) - np.min(qs))
    ps = ps / np.sum(ps)

    agent_obs = [ob[agent_idx] for ob in obss]
    agent_acs = [ac[agent_idx] for ac in acts]

    # step 2: sample points 
    if is_reweight:
        idx = np.random.choice(len(agent_obs), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        idx = np.random.choice(
            len(obss), size=min(max_pts, np.sum(ps > 0), replace=False)
        )
    # step 3: obtain sampled indices
    # TODO: make sure that we are indexing properly for returning 
    return agent_obs[idx], agent_acs[idx], qs[idx]


def _sample(obss, acts, qs, max_pts, is_reweight, addtl_qs=None, direct_use=False):
    # Step 1: Compute probabilities

    # THIS IS SUPPOSED TO BE CORRECT
    if len(qs.shape) == 3:
        qs = qs[0]
    critical_state_metric = 'new'
    combine_qs = True
    if combine_qs:
        if addtl_qs is not None:
            new_qs = np.sum((qs, addtl_qs), axis=0)
            assert new_qs.shape == qs.shape
            qs = new_qs

    if critical_state_metric == 'second_best':
        ps = np.max(qs, axis=1) - np.partition(qs, -2)[:, -2]
    elif not direct_use:
        ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    else:
        fn = lambda x: np.maximum(x, 0)
        ps = fn(qs[0])
    print('ps shape is', ps.shape)
    print("The maximum p is ", np.max(ps))
    print("The minimum ps is ", np.min(ps))
    ps = ps / np.sum(ps)
    print('len obs is ', len(obss))
    print('ps is', len(ps))

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(
            len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False
        )

        # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], ps[idx]


def _sample_joint(obss, acts, qs, max_pts, is_reweight, train_frac=0.8, direct_use_qs=False):
    # Split the data into train set and test set at this stage
    # Need to be done before resampling

    obss_train, acts_train, obss_test, acts_test, train_idx, test_idx = split_train_test(obss, acts, train_frac,
                                                                                         return_index=True)

    ret_obss_train, ret_acts_train = [], []
    ret_obss_test, ret_acts_test = [], []

    for agent in range(len(qs)):
        qs_train, qs_test = qs[agent][train_idx], qs[agent][test_idx]

        if direct_use_qs:
            fn = lambda x: np.maximum(x, 0)
            prob_train, prob_test = fn(qs_train), fn(qs_test)
        else:
            prob_train = np.max(qs_train, axis=1) - np.min(qs_train, axis=1)
            prob_test = np.max(qs_test, axis=1) - np.min(qs_test, axis=1)

        prob_train = prob_train / np.sum(prob_train)
        prob_test = prob_test / np.sum(prob_test)

        if is_reweight:
            idx_train = np.random.choice(len(obss_train), size=min(max_pts, np.sum(prob_train > 0)), p=prob_train)
            idx_test = np.random.choice(len(obss_test), size=min(max_pts, np.sum(prob_test > 0)), p=prob_test)
        else:
            idx_train = np.random.choice(len(obss_train), size=min(max_pts, np.sum(prob_train > 0)), replace=False)
            idx_test = np.random.choice(len(obss_test), size=min(max_pts, np.sum(prob_test
                                                                                 > 0)), replace=False)
        ret_obss_train.append(obss_train[idx_train])
        ret_acts_train.append(acts_train[idx_train])
        ret_obss_test.append(obss_test[idx_test])
        ret_acts_test.append(acts_test[idx_test])
    return ret_obss_train, ret_acts_train, ret_obss_test, ret_acts_test


class TransformerPolicy:
    def __init__(self, policy, state_transformer, easy_feature=False):
        self.policy = policy
        self.state_transformer = state_transformer
        self.easy_feature = easy_feature

    def predict(self, obss, info=None):
        if self.easy_feature:
            obss = [info]
        return self.policy.predict(
            np.array([self.state_transformer(obs) for obs in obss])
        )


def test_policy(env, policy, state_transformer, n_test_rollouts, index, env_name, global_seed,
                selection_criteria, max_timesteps=25):
    if isinstance(policy, DTPolicy):
        wrapped_student = TransformerPolicy(policy, state_transformer)
    else:
        wrapped_student = policy
    cum_rew = 0.0
    rews, infos, other_infos = [], [], []

    for i in range(n_test_rollouts):
        seed = i * global_seed
        student_trace = get_rollout(
            env=env,
            policy=wrapped_student,
            render=False,
            scenario_name=env_name,
            seed=seed
        )
        if isinstance(student_trace[0][4], list):
            rew = sum((np.sum(np.array(rew)[index]) for _, _, _, _, rew, _ in student_trace))
        else:
            rew = sum((np.sum([rew]) for _, _, _, _, rew, _ in student_trace))
        cum_rew += rew
        rews.append(rew)
        new_infos = [inf for _, _, _, _, _, inf in student_trace]
        infos.append(new_infos)
        other_infos.extend(new_infos)

    avg_rew = cum_rew / n_test_rollouts
    rew_var = np.var(np.array(rews))
    if selection_criteria == 'score' and env_name == 'simple_adversary':
        score, score_var = compute_successes(
            other_infos,
            n_test_rollouts,
            max_timesteps,
            env_name=env_name
        )
        if index != -1:
            return (score[index], score_var[index]), avg_rew, infos
        else:
            return (score, score_var), avg_rew, infos
    else:
        return (avg_rew, rew_var), avg_rew, infos


# TODO: have generalized way to get successes by iterating through
# and saying hey this is the avg succ per agent given the infos 
def compute_successes(infos, n_test_rollouts, n_timesteps, env_name='simple_spread', epsilon=0.1):
    num_agents = len(infos[0]['n'])
    if env_name == 'simple_adversary':
        agent_successes = [[] for _ in range(len(infos[0]['n']))]
        for it in infos:
            # if it['n'][0][2] < epsilon:
            if it['n'][0][2] < epsilon:
                agent_successes[0].append(int(True))
            else:
                agent_successes[0].append(int(False))

            if (it['n'][1][0] < epsilon and it['n'][2][1] < epsilon) \
                    or (it['n'][1][1] < epsilon and it['n'][2][0] < epsilon):
                agent_successes[1].append(int(True))
                agent_successes[2].append(int(True))
            else:
                agent_successes[1].append(int(False))
                agent_successes[2].append(int(False))
        agent_successes = np.array(agent_successes).reshape(
            (num_agents, n_test_rollouts, n_timesteps)
        )

        ag_succs = [np.sum(ag_s, axis=1) > 0 for ag_s in agent_successes]
        avg_succ = np.average(np.array(ag_succs), axis=1)
        succ_var = np.var(np.array(ag_succs), axis=1)
    elif env_name == 'simple_spread':
        landmarks_info = [[] for _ in range(num_agents)]
        for ep in infos:
            print('ep is', ep)
            agent_landmarks = [[] for _ in range(num_agents)]
            for it in ep:
                print('it is', it)
                for a_i, agent_info in enumerate(it['n']):
                    agent_landmarks[a_i].append(int(agent_info[3] > 0))
            # print(len(collision_info))
            for a_i in range(nagents):
                landmarks_info[a_i].append(agent_landmarks[a_i])
        avg_succ = np.mean(np.sum(np.array(landmarks_info), axis=2), axis=1)
        succ_var = np.var(np.sum(np.array(landmarks_info), axis=2), axis=1)
    return (avg_succ, succ_var)


def compare_policies(env, other_pols, expert_pols, state_transformer, n_test_rollouts, global_seed):
    env_name = 'simple_adversary'
    wrapped_experts, wrapped_students = [], []
    for expert_pol in expert_pols:
        if isinstance(expert_pol, DTPolicy):
            wrapped_expert = TransformerPolicy(expert_pol, state_transformer)
        else:
            wrapped_expert = expert_pol
        wrapped_experts.append(wrapped_expert)

    for other_pol in other_pols:
        if isinstance(other_pol, DTPolicy):
            wrapped_student = TransformerPolicy(other_pol, state_transformer)
        else:
            wrapped_student = other_pol
        wrapped_students.append(wrapped_student)

    diffs = []
    for i in range(n_test_rollouts):
        expert_trace = get_joint_rollout(
            env, wrapped_experts, False, env_name, global_seed * i
        )
        assert len(expert_trace) == 25
        episode_diffs = [[] for _ in range(len(wrapped_experts))]
        for s, s_all, a, a_all, rew, infos in expert_trace:
            for w, wrapped_student in enumerate(wrapped_students):
                pred_act = np.array(wrapped_student.predict([s[w]])[0])
                # TODO: get simulated next state?
                actual_act = np.array(a_all[w])
                assert pred_act.shape == actual_act.shape
                episode_diffs[w].append(int(np.array_equal(pred_act, actual_act)))
        diffs.append(episode_diffs)
    diffs = np.array(diffs)
    per_timestep_avg = np.mean(np.mean(diffs, axis=1), axis=0)
    # print('per timestep avg', per_timestep_avg)
    # print('diffs sum', np.sum(diffs))
    # print(per_timestep_avg.shape)
    assert per_timestep_avg.shape[0] == 25  # n_test_rollouts
    per_agent_avg = np.mean(np.mean(diffs, axis=2), axis=0)
    # ('per agent avg', per_agent_avg)
    assert per_agent_avg.shape[0] == len(wrapped_experts)
    per_agent_timestep_avg = np.mean(diffs, axis=0)
    return per_timestep_avg, per_agent_avg, per_agent_timestep_avg


def test_joint_policies(env, policies, state_transformer, n_test_rollouts, env_name, global_seed, selection_criteria):
    wrapped_students = [
        TransformerPolicy(p, state_transformer) if isinstance(p, DTPolicy) else p for p in policies
    ]
    cum_rew = np.array([0.0 for _ in range(len(policies))])
    rews, infos, other_infos = [], [], []
    for i in range(n_test_rollouts):
        student_trace = get_joint_rollout(env, wrapped_students, False, env_name, seed=i * global_seed)
        rollout_rews = np.array([rew for _, _, _, _, rew, _ in student_trace])
        sum_rollout_rews = np.sum(rollout_rews, axis=0)
        cum_rew += sum_rollout_rews
        rews.append(sum_rollout_rews)
        # cum_rew += sum((rew for _, _, _, _, rew, _ in student_trace))
        # rews.append(sum((rew for _, _, _, _, rew, _ in student_trace)))
        new_infos = [inf for _, _, _, _, _, inf in student_trace]
        infos.append(new_infos)
        other_infos.extend(new_infos)
    avg_rew = cum_rew / n_test_rollouts
    rew_var = np.var(np.array(rews), axis=0)

    if env_name == 'simple_adversary' and selection_criteria == 'score':
        return other_infos, avg_rew, infos
    else:
        return avg_rew, rew_var, infos

    # TODO: how to identify the best policies when there is a mixed comp coop scenario?


def identify_best_policies(env, all_policies, state_transformer, n_test_rollouts):
    # cut policies in half on each iteration 
    for i, policies in enumerate(all_policies):
        # Step 1: Sort policies by current estimated reward 
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies 
        n_policies = int((len(policies) + 1) / 2)
        all_policies[i] = policies

    # Step 3: Build new policies 
    new_policies = []
    for i in range(n_policies):
        policy, rew = policies[i]
        new_rew, _, _ = test_policy(env, )
        new_policies.append((policy, new_rew))
    all_policies = anew_policies

    if len(policies) != 1:
        raise Exception

    return [policies[0][0] for policies in all_policies]


def identify_best_joint_policies(
        env, policies, state_transformer, n_test_rollouts, selection_criteria, team_indices,
        env_name='simple_adversary', n_init_pol=10, global_seed=1
):
    # NOTE: we removed the halfing procedure. you can either add it back in by looping through and halfing
    # or the same effect can be obtained by doing a large number of environment rollouts by setting 
    # n_test_rollouts to some large value 

    # if we don't constrain the number of policies used for mix and match, then just use all of them 
    if n_init_pol is not None:
        # if we have somehow misspecified the number of policies, update that
        n_policies = []
        for p, pol in enumerate(policies):
            if p not in team_indices:
                n_policies.append(1)
            else:
                if n_init_pol > len(pol):
                    n_policies.append(len(pol))
                else:
                    n_policies.append(n_init_pol)

        # sort the policies and grab the best-performing ones
        policies = [
            sorted(agent_policies, key=lambda entry: -entry[1]) if a in team_indices else agent_policies for
            a, agent_policies in enumerate(policies)
        ]
        policies = [pol[:n_policies[i]] if i in team_indices else pol for i, pol in enumerate(policies)]

    policy_table = [[] for _ in range(len(team_indices))]  # hardcoded number of agents for now
    # first do mix and match on all of the policies -- not pretty code 
    # TODO: update to be more general for greater than two agent teams 
    other_agent_indices = [k for k in range(len(policies)) if k not in team_indices]
    joint_policy = [None for _ in range(len(policies))]
    for i in range(len(policies[team_indices[0]])):
        for j in range(len(policies[team_indices[1]])):
            a0pol, a1pol = team_indices[0], team_indices[1]
            agent1_policy, rew1, rew_var1, a1_idx = policies[a0pol][i]
            agent2_policy, rew2, rew_var2, a2_idx = policies[a1pol][j]

            # TODO: come up with a better way to implement this 
            if len(other_agent_indices) > 0:
                for k in range(len(policies)):
                    if k in other_agent_indices:
                        joint_policy[k] = policies[k]
                joint_policy[team_indices[0]] = agent1_policy
                joint_policy[team_indices[1]] = agent2_policy
            else:
                joint_policy = [agent1_policy, agent2_policy]

            infos, addtl_info, true_infos = test_joint_policies(
                env, joint_policy, state_transformer, n_test_rollouts, selection_criteria=selection_criteria,
                env_name=env_name,
                global_seed=global_seed
            )
            if selection_criteria == 'score':
                agent_succs, agent_vars = compute_successes(infos, n_test_rollouts, 25, env_name=env_name)
            else:
                agent_succs, agent_vars = infos, addtl_info
            policy_table[0].append(
                (agent1_policy, agent_succs[team_indices[0]], agent_vars[team_indices[0]], a1_idx))  # , agent_vars[1]))
            policy_table[1].append(
                (agent2_policy, agent_succs[team_indices[1]], agent_vars[team_indices[1]], a2_idx))  # agent_vars[1]]))
    # sort the policies 
    best_pols = [
        sorted(pol, key=lambda entry: -entry[1]) for pol in policy_table
    ]
    # TODO: generalize to > 2 agents per team 
    print(best_pols[0][0])
    print(best_pols[1][0])
    return [best_pols[0][0], best_pols[1][0]]


def identify_best_joint_policy(policies, parameters, team):
    policies = policies if parameters.test_only == -1 else policies[parameters.test_only - 1:parameters.test_only]
    best_policy, best_policy_info = [None] * len(policies[0][0]), [None] * len(policies[0][0])
    for t in set(team.values()):
        # Select best agent for team t
        idx = [i for i in range(len(policies[0][0])) if team[i] == t]
        best_score, best_t = None, None
        for i, policy in enumerate(policies):
            if len(policy) == 0:
                continue
            score = sum(policy[1][j] for j in idx)
            if best_score is None or score > best_score:
                best_score = score
                for j in idx:
                    best_policy[j] = policy[0][j]
                    best_policy_info[j] = (policy[1][j], policy[2][j], i)
    print(f'Chose policy with score {best_policy_info}')
    return best_policy, best_policy_info


def identify_best_policy(env, policies, state_transformer, n_test_rollouts, teacher_index, global_seed=1,
                         env_name='simple_adversary'):
    # log('Initial policy count: {}'.format(len(policies)), INFO)
    # cut policies by half on each iteration
    # NOTE: again, halfing code is commented out. should still work. we were just simulating 
    # with a larger number of rollouts in the environment.
    print('Initial policy count: {}'.format(len(policies)))
    initial_policies = sorted(policies, key=lambda entry: -entry[1])
    print('initial policy evals are', initial_policies[:10])

    '''
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        b_pol, b_score, b_var, idx = policies[0]
        if b_score > best_score_so_far:
            best_score_so_far = b_score
            best_pol_var = b_var 
            best_policy_so_far = b_pol
            best_idx = idx 
        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew, rew_var, idx = policies[i]
            new_rew, infos, new_infos = test_policy(
                env=env, 
                policy=policy, 
                state_transformer=state_transformer, 
                n_test_rollouts=n_test_rollouts, 
                teacher_index=teacher_index, 
                env_name=env_name,
                global_seed=global_seed
            )
            score, score_var = new_rew 
            if isinstance(new_rew[0], list):
                new_policies.append((policy, score[teacher_index], score_var[teacher_index], idx))
            else:
                new_policies.append((policy, score, score_var, idx))
            print('Score update ({}): {} -> {}'.format(idx, (rew, rew_var), new_rew))

        policies = new_policies

    if len(policies) != 1:
        raise Exception()
    '''
    policies = initial_policies[0]
    print('current best policy score', policies[1])
    print('chosen idx', policies[3])
    '''
    # now let's compare the two that we have
    test_rew, test_infos, test_new_infos = test_policy(
        env, policies[0][0], state_transformer, 1000, teacher_index 
    )
    print('score for chosen is: ', test_rew)
    best_rew, best_infos, best_new_infos = test_policy(
        env, best_policy_so_far, state_transformer, 1000, teacher_index
    )
    print('score for best is: ', best_rew)
    '''
    return policies


def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i + seq_len])
    return seqs


def get_action_sequences(env, policy, seq_len, n_rollouts):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, False)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])
    return seqs_sorted


def parse_obs_acts(student_trace, n_agents, teachers=None):
    all_obss, all_acss = [], []
    for i in range(n_agents):
        all_obss.append([])
        all_acss.append([])
        if len(student_trace[0]) == 4:
            for obs, acs, _, _ in student_trace:
                all_obss[i].append(obs[i])
                all_acss[i].append(acs[i])
        elif len(student_trace[0]) == 6:
            for obs, _, acs, _, _, _ in student_trace:
                all_obss[i].append(obs[i])
                assert teachers is not None, "teachers must be provided since the action space might be different for each agent"
                start = sum(teachers[j].num_actions for j in range(i))
                all_acss[i].append(acs[start: start + teachers[i].num_actions])
    return all_obss, all_acss


def train_dt(student, obss, acts, qs, max_samples, train_frac, is_reweight):
    cur_obss, cur_acts, cur_qs = _sample(
        np.array(obss), np.array(acts), np.array(qs), max_samples, is_reweight
    )
    student.train(cur_obss, cur_acts, train_frac, )
    return student


def create_masks(state_transform, num_mask, teacher_index):
    num_mask = 1
    if teacher_index == 1:
        pass
        # state_transform[2] = num_mask
        # state_transform[3] = num_mask
    if teacher_index == 2:
        pass
        # state_transform[0] = num_mask
        # state_transform[1] = num_mask
    if teacher_index == 1 or teacher_index == 2:
        # state_transform[6] = 0
        # state_transform[7] = 0
        state_transform[8] = num_mask
        state_transform[9] = num_mask

    return state_transform


def train_dagger_joint(
        env, teacher, student, state_transformer, max_iters, n_batch_rollouts,
        max_samples, train_frac, is_reweight, n_test_rollouts,
        save_folder, save_file_name, parameters, env_name='simple_adversary', selection_criteria='success'
):
    student_folder = save_folder + 'agent/'
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    # Step 0: Setup
    np.random.seed(parameters.random_seed)
    random_seed = np.random.randint(0, 10000)
    obss, all_obs, acts, all_acts, qs, infos = [], [], [], [], [], []
    students, student_data = [], {}
    wrapped_student = TransformerPolicy(student, state_transformer,
                                        easy_feature=env_name == 'simple_tag' and parameters.easy_feature)

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts * parameters.warmup, env_name, global_seed=random_seed)
    obss.extend((state_transformer(obs) for obs, _, _, _, _, _ in trace))
    acts.extend((act for _, _, act, _, _, _ in trace))
    infos.extend((info for _, _, _, _, _, info in trace))
    if env_name == 'simple_tag' and parameters.easy_feature:
        env.easy_feature_sizes = []
        for i in infos[0]['n']:
            env.easy_feature_sizes.append(len(i[1]))

    cuda = False
    if not cuda:
        cast = lambda x: Variable(Tensor(x), requires_grad=False)
    else:
        cast = lambda x: Variable(torch.from_numpy(x).cuda(), requires_grad=False)

    new_obs = [cast(np.concatenate(ob)) for ob in obss]
    new_acts = [cast(np.array(ac)) for ac in acts]
    new_infos = []
    if env_name == 'simple_tag' and parameters.easy_feature:
        new_infos = [cast(np.concatenate([agent_info[-1] for agent_info in info['n']])) for info in infos]

    all_obs, all_acts, all_infos = new_obs, new_acts, new_infos
    all_qs = []

    # Calculate each agent's Q-value for the expert rollout
    team = student.agent_team
    sa_envs = [SingleAgentWrapper(env, teacher.agents, t_i, False, parameters.scenario_name) for t_i in
               range(teacher.nagents)]
    teachers = [TeacherFunc(sa_envs[i], teacher.agents[i], i, team=team) for i in range(teacher.nagents)]
    parsed_obs, parsed_acs = parse_obs_acts(trace, teacher.nagents, teachers)
    parsed_new_obss = [cast(np.array(ob)) for ob in parsed_obs]
    parsed_new_acts = [cast(np.array(ac)) for ac in parsed_acs]

    for i in range(teacher.nagents):
        all_qs.append(1 * (
            teachers[i].predict_average_others(parsed_new_obss, parsed_new_acts) if parameters.average_others
            else (
                teachers[i].predict_optimal_others(parsed_new_obss, parsed_new_acts) if parameters.optimal_others
                else (
                    teachers[i].predict_max_min_minus_min_max(parsed_new_obss, parsed_new_acts) if parameters.maxmin
                    else teachers[i].iviper_resampling(parsed_new_obss, parsed_new_acts)
                )
            )
        ))

    # Step 2: Dagger outer loop
    for i in tqdm(range(max_iters)):
        random_seed = np.random.randint(0, 10000)

        # Step 2a: Train from a random subset of aggregated data
        obs_to_use = all_obs
        if env_name == 'simple_tag' and parameters.easy_feature:
            obs_to_use = all_infos

        cur_obss_train, cur_acts_train, cur_obss_test, cur_acts_test = _sample_joint(
            obss=np.array(torch.stack(obs_to_use)),
            acts=np.array(torch.stack(all_acts)),
            qs=all_qs,
            max_pts=max_samples,
            is_reweight=is_reweight,
            train_frac=train_frac,
            direct_use_qs=True,
        )
        # store the data that is used to train each individual dt policy
        if i not in student_data.keys():
            student_data[i] = []
        student_data[i].append([cur_obss_train, cur_acts_train, cur_obss_test, cur_acts_test])

        start = time.time()
        student.train(cur_obss_train, cur_acts_train, cur_obss_test, cur_acts_test, parameters=parameters)
        print('Time to train student: {}'.format(time.time() - start))

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, wrapped_student, False, n_batch_rollouts, env_name,
                                     global_seed=random_seed)

        student_obss, student_acts, student_infos = [], [], []
        student_obss.extend((state_transformer(obs) for obs, _, _, _, _, _ in student_trace))
        student_acts.extend((act for _, _, act, _, _, _ in student_trace))
        student_infos.extend((info for _, _, _, _, _, info in student_trace))

        new_obs = [cast(np.concatenate(ob)) for ob in student_obss]
        new_acts = [cast(np.array(ac)) for ac in student_acts]
        if env_name == 'simple_tag' and parameters.easy_feature:
            new_infos = [cast(np.concatenate([agent_info[-1] for agent_info in info['n']])) for info in student_infos]
        else:
            new_infos = []

        # Step 2c: Query the oracle for supervision
        for k, obs in enumerate(new_obs):
            # Temporarily change the easy_feature flag to False
            student.parameters.easy_feature = False
            new_acts[k] = torch.cat(teacher.predict(student._parse(obs.view(1, -1), t='obs')), axis=1).view(-1).detach()
            student.parameters.easy_feature = parameters.easy_feature
        teacher_qs = []

        # modify student trace by replacing action with that of the teacher
        for k in range(len(student_trace)):
            student_trace[k] = list(student_trace[k])
            student_trace[k][2] = new_acts[k].detach().numpy()
        parsed_obs, parsed_acs = parse_obs_acts(student_trace, teacher.nagents, teachers=teachers)
        parsed_new_obss = [cast(np.array(ob)) for ob in parsed_obs]
        parsed_new_acts = [cast(np.array(ac)) for ac in parsed_acs]

        for agent in range(teacher.nagents):
            teacher_qs.append(
                teachers[agent].predict_average_others(parsed_new_obss, parsed_new_acts) if parameters.average_others
                else (
                    teachers[agent].predict_optimal_others(parsed_new_obss,
                                                           parsed_new_acts) if parameters.optimal_others
                    else (
                        teachers[agent].predict_max_min_minus_min_max(parsed_new_obss,
                                                                      parsed_new_acts) if parameters.maxmin
                        else teachers[agent].iviper_resampling(parsed_new_obss, parsed_new_acts)
                    )
                )
            )

        # Step 2d: Add the augmented state-action pairs back to aggregate
        all_obs.extend(new_obs)
        all_acts.extend(new_acts)
        all_infos.extend(new_infos)
        for agent in range(teacher.nagents):
            all_qs[agent] = np.concatenate((all_qs[agent], teacher_qs[agent]), axis=0)

        # Step 2e: Estimate the reward
        cur_rew = np.sum(np.stack([rew for _, _, _, _, rew, _ in student_trace]), axis=0) / n_batch_rollouts

        if selection_criteria == 'success' or env_name == 'simple_adversary':
            infos = [info for _, _, _, _, _, info in student_trace]
            curr_succ = compute_successes(
                infos, n_batch_rollouts, 25,
                env_name=env_name
            )  # [teacher_index]
            score, score_var = curr_succ
            print(f'Current success rate: {score}, current reward: {cur_rew}')
            students.append((student.make_individual(), score, score_var, cur_rew))
        else:
            print(f'Current reward: {cur_rew}')
            students.append((student.make_individual(), cur_rew))

        student_folder = save_folder + 'agent'
        save_dt_policy(students[-1], f'{student_folder}', save_file_name + '_' + str(i) + '.pk')

    return students, student_data


def calc_exploitability(agents, env, team_info, save_dir, fast=0, print_results=True):
    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'

    ret = []
    for team in set(team_info.values()):
        team_agents_idx = [agent for agent in range(len(agents)) if team_info[agent] == team]
        if print_results:
            print(f'Calculating exploitability for team {team} (composed of {team_agents_idx})')
        existing_agents = dict()
        for idx in team_agents_idx:
            existing_agents[idx] = agents[idx]
        eval_env = ExploitabilityWrapper(env, existing_agents)
        # Train exploit agent using maddpg, this is different from main.py is that eval_env is not vectorized
        maddpg = MADDPG.init_from_env(eval_env)
        replay_buffer = ReplayBuffer(max_steps=int(1e6),
                                     num_agents=maddpg.nagents,
                                     obs_dims=[obsp.shape[0] for obsp in eval_env.observation_space],
                                     ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                              for acsp in eval_env.action_space])

        fast = int(fast)
        config = {
            'seed': 123,
            'n_episodes': 10000 // (1 + fast),
            'n_exploration_eps': 5000 // (1 + fast),
            'init_noise_scale': 1.0,
            'final_noise_scale': 0.05,
            'episode_length': 25,
            'batch_size': 1024 * 25,
            'steps_per_update': 100,
        }
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.set_num_threads(6)
        steps = 0
        rews_history = []
        for ep_i in tqdm(range(0, config['n_episodes'])):
            maddpg.prep_rollouts(device=device)
            explr_pct_remaining = max(0, config['n_exploration_eps'] - ep_i) / config['n_exploration_eps']
            maddpg.scale_noise(
                config['final_noise_scale'] + (
                        config['init_noise_scale'] - config['final_noise_scale']) * explr_pct_remaining)
            maddpg.reset_noise()
            obs = eval_env.reset()
            done = False
            t = 0
            reward = 0
            while not done:
                torch_obs = [
                    Variable(torch.tensor(obs[i], device=device, dtype=torch.float).view(1, -1), requires_grad=False)
                    for i in eval_env.get_missing_agent()]
                all_obs = [
                    Variable(torch.tensor(obs[i], device=device, dtype=torch.float).view(1, -1), requires_grad=False)
                    for i in range(eval_env.total_agents)]
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
                if device == 'cuda':
                    agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]
                else:
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                next_obs, rewards, dones, infos = eval_env.step(agent_actions, all_obs, device=device)
                t += 1
                steps += 1
                done = bool(sum(dones))
                replay_buffer.push(np.array([[obs[i] for i in eval_env.get_missing_agent()]]),
                                   agent_actions,
                                   np.array([[rewards[i] for i in eval_env.get_missing_agent()]]),
                                   np.array([[next_obs[i] for i in eval_env.get_missing_agent()]]),
                                   np.array([[dones[i] for i in eval_env.get_missing_agent()]]),
                                   cur_ep=ep_i)
                reward += np.mean([[rewards[i] for i in eval_env.get_missing_agent()]])
                obs = next_obs

                if len(replay_buffer) >= config['batch_size'] and steps % config['steps_per_update'] == 0:
                    maddpg.prep_training(device=device)
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config['batch_size'], to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i)
                    maddpg.update_all_targets()
                    maddpg.prep_rollouts(device=device)

                if t >= config['episode_length']:
                    break

            ep_rews = replay_buffer.get_total_rewards(config['episode_length'])
            rews_history.append(reward)

        if print_results:
            print('team is', team)
        # team_info_dict[str(team)] = rews_history
        ret.append(float('%.2f' % np.array(rews_history[-100:]).mean()))
        if print_results:
            print('Maximum opponent reward', np.array(rews_history[-100:]).mean())

    for agent in agents:
        if isinstance(agent, DDPGAgent):
            agent.policy.to(device='cpu')

    return ret


def train_dagger(
        env, teachers, student, state_transformer, max_iters, n_batch_rollouts,
        max_samples, train_frac, is_reweight, n_test_rollouts, teacher_index,
        save_folder, save_file_name, parameters, env_name='simple_adversary', selection_criteria='success'
):
    use_addtl = False
    student_folder = save_folder + 'agent' + str(teacher_index) + '/'
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
    # Step 0: Setup
    np.random.seed(parameters.random_seed)
    random_seed = np.random.randint(0, 10000)
    obss_for_student, all_obs, acts, all_acts, qs = [], [], [], [], []
    students, student_data = [], {}
    wrapped_student = TransformerPolicy(student, state_transformer)

    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teachers[teacher_index], False, n_batch_rollouts * parameters.warmup, env_name,
                         global_seed=random_seed)
    if 'SingleAgentWrapper' in str(type(env)):
        o_shapes = env.env.observation_space
    else:
        o_shapes = env.observation_space
    state_transform = np.ones(o_shapes[teacher_index].shape)
    state_transform = create_masks(state_transform, 1, teacher_index)

    obs_to_store = []
    # for obs, _, _, _, _, _ in trace:
    #    print(state_transform * obs)
    obs_to_store.extend(state_transform * obs for obs, _, _, _, _, _ in trace)
    obss_for_student = obs_to_store
    # obss.extend((state_transformer(obs) for obs, _, _, _, _, _ in trace))
    acts.extend((act for _, _, act, _, _, _ in trace))

    for i in range(len(teachers)):
        all_obs.append([])
        all_acts.append([])
        for _, obs, _, acs, _, _ in trace:
            all_obs[i].append(obs[i])
            all_acts[i].append(acs[i])

    cast = lambda x: Variable(Tensor(x), requires_grad=False)
    new_obs = [cast(np.array(ob)) for ob in all_obs]
    new_acts = [cast(np.array(ac)) for ac in all_acts]
    qs = [teacher.predict_q_for_each_action(new_obs, new_acts) for teacher in teachers]

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        # Step 2a: Train from a random subset of aggregated data
        random_seed = np.random.randint(0, 10000)
        if use_addtl and teacher_index > 0:
            addtl_qs = qs[1] if teacher_index == 2 else qs[2]
        else:
            addtl_qs = None
        cur_obss, cur_acts, cur_qs = _sample(
            obss=np.array(obss_for_student),
            acts=np.array(acts),
            qs=np.array(qs[teacher_index]),
            max_pts=max_samples,
            is_reweight=is_reweight,
            addtl_qs=addtl_qs
        )

        student_train_info = student.train(cur_obss, cur_acts, train_frac, )

        # store info about each individual dt policy
        student_data[i] = {}
        train_accuracy, test_accuracy, num_nodes, feature_importances = student_train_info
        student_data[i]['train accuracy'] = train_accuracy
        student_data[i]['test accuracy'] = test_accuracy
        student_data[i]['feature importances'] = feature_importances
        student_data[i]['num nodes'] = num_nodes

        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, wrapped_student, False, n_batch_rollouts, env_name, global_seed=random_seed)
        student_obss = [obs for obs, _, _, _, _, _ in student_trace]
        all_obss, all_acss = [], []
        for j in range(len(teachers)):  # TODO: hardcoded
            all_obss.append([])
            all_acss.append([])
            for _, obs, _, acs, _, _ in student_trace:
                all_obss[j].append(obs[j])
                all_acss[j].append(acs[j])
        new_obs = [cast(np.array(ob)) for ob in all_obss]
        new_acts = [cast(np.array(ac)) for ac in all_acss]

        # Step 2c: Query the oracle for supervision
        teacher_qs = [teacher.predict_q_for_each_action(
            new_obs, new_acts
        ) for teacher in teachers]  # at the interface level, order matters, since teacher.predict may run updates
        obss_for_student.extend(state_transform * student_obss)
        torch_student_obs = cast(np.array(student_obss))
        teacher_acts = teachers[teacher_index].predict(torch_student_obs)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        all_obs.extend(all_obss)
        all_acts.extend(all_acss)
        acts.extend(teacher_acts.data.numpy())
        qs = [np.concatenate((q, teacher_q)) for q, teacher_q in zip(qs, teacher_qs)]

        # save the dt policy to recover later
        student_folder = save_folder + 'agent' + str(teacher_index) + '/'
        save_dt_policy(
            student, student_folder, save_file_name + '_' + str(i) + '.pk'
        )

        # Step 2e: Estimate the reward
        rews = [rew for _, _, _, _, rew, _ in student_trace]
        cur_rew = sum(rews) / n_batch_rollouts
        cur_rew_var = np.var(np.array(rews))

        if selection_criteria == 'success':
            infos = [info for _, _, _, _, _, info in student_trace]
            curr_succ = compute_successes(
                infos, n_batch_rollouts, 25, env_name=env_name
            )  # [teacher_index]
            score, score_var = curr_succ
            students.append((student.clone(), score[teacher_index], score_var[teacher_index]))
        else:
            students.append((student.clone(), cur_rew, cur_rew_var))
    print('students are', students)
    return students, student_data
