import torch
import pickle
import pandas as pd
import numpy as np
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from multiagent.scenarios.simple_cybersec import CybersecScenario
from util.env_wrappers import SingleAgentWrapper
from core.rl import *
from core.dt import *
from torch.autograd import Variable
import os

from algorithms.maddpg import MADDPG
from maviper.python.viper.core.dt import DTPolicy
from maviper.python.viper.core.joint_dt import JointDTPolicy
from maviper.python.viper.core.rl import calc_exploitability
from utils.agents import DDPGAgent, FittedQDTAgent, ImitatedDTAgent


def calc_success_info(info, scenario, epsilon=0.1):
    assert scenario == 'simple_adversary', 'calc success info not implemented for scenarios other than simple_adversary'
    adv_distance_to_t0, adv_distance_to_t1, adv_distance_to_target = info['n'][0]
    adversary_success = adv_distance_to_target < epsilon
    agent1_target0, agent1_target1, agent1_goal_target = info['n'][1]
    agent2_target0, agent2_target1, agent2_goal_target = info['n'][2]
    # Agents cover both targets
    if (agent1_target0 < epsilon and agent2_target1 < epsilon) \
            or (agent1_target1 < epsilon and agent2_target0 < epsilon):
        agent_success = True
    else:
        agent_success = False
    return adversary_success, agent_success


def log_success_percent(infos, prefix, epsilon, label_info, scenario_name, print_result=True):
    if scenario_name == 'simple_adversary':
        agent0_successes, agent1_successes, agent0_s, agent1_s = [], [], [], []
        both_close, neither_close, one_close = [], [], []
        adversary_both_targets, adversary_wrong_target = [], []
        n_tsteps = 25
        for ep in infos:
            agent0_ep_successes, agent1_ep_successes = [], []
            adversary_wrong = []
            bc, nc, oc = [], [], []
            assert len(ep) == n_tsteps
            # data consists of adversary dist, (agent_dists to landmarks, goal)
            for it in ep:
                # check the adversary distance 
                # adv_distance_to_target = it['n'][0]
                adv_distance_to_t0, adv_distance_to_t1, adv_distance_to_target = it['n'][0]
                if adv_distance_to_target < epsilon:
                    agent0_ep_successes.append(int(True))
                else:
                    agent0_ep_successes.append(int(False))

                if adv_distance_to_target != adv_distance_to_t0:
                    other_target = adv_distance_to_t0
                else:
                    other_target = adv_distance_to_t1
                if other_target < epsilon:
                    adversary_wrong.append(int(True))
                else:
                    adversary_wrong.append(int(False))

                # now we look at the agent behavior 
                # if one of the agents is close to the first target
                # and the other agent is close to the other target and vice versa
                agent1_target0, agent1_target1, agent1_goal_target = it['n'][1]
                agent2_target0, agent2_target1, agent2_goal_target = it['n'][2]
                if (agent1_target0 < epsilon and agent2_target1 < epsilon) \
                        or (agent1_target1 < epsilon and agent2_target0 < epsilon):
                    agent1_ep_successes.append(int(True))
                else:
                    agent1_ep_successes.append(int(False))

                # if both go to same target 
                if (agent1_target0 < epsilon and agent2_target0 < epsilon) \
                        or (agent1_target1 < epsilon and agent2_target1 < epsilon):
                    bc.append(int(True))
                else:
                    bc.append(int(False))
                if (agent1_target0 >= epsilon and agent1_target1 >= epsilon) \
                        or (agent2_target0 >= epsilon and agent2_target1 >= epsilon):
                    nc.append(int(True))
                else:
                    nc.append(int(False))
            adversary_wrong_target.append(int(sum(adversary_wrong) > 0))
            # print('adv wrong target', adversary_wrong_target)
            adversary_both_targets.append(int((sum(adversary_wrong) > 0) and sum(agent0_ep_successes) > 0))
            # print('adversary both targets', adversary_both_targets)
            agent0_successes.append(agent0_ep_successes)
            # print('adversary ep succes', agent0_ep_successes)
            agent0_s.append(int(sum(agent0_ep_successes) > 0))
            agent1_successes.append(agent1_ep_successes)
            agent1_s.append(int(sum(agent1_ep_successes) > 0))

            both_close.append(int(sum(bc) > 0))
            neither_close.append(int(sum(nc) > 0))

        labels = ['Adversary', 'Agents']
        successes = [
            sum(agent0_s) / len(agent0_s),
            sum(agent1_s) / len(agent1_s)
        ]
        # print('both targets: ', np.average(np.array(adversary_both_targets)))
        # print('wrong target: ', np.average(np.array(adversary_wrong_target)))
        # print('correct target: ', np.average(np.array(agent0_successes)))

        # print('both close: ', np.average(np.array(both_close)))
        # print('neither close: ', np.average(np.array(neither_close)))

        success_var0 = np.var(np.array(agent0_s))
        success_var1 = np.var(np.array(agent1_s))
        # print('vars:', (success_var0, success_var1))
        success_dict = {
            'Agent': labels,
            'Success %': successes,
            'Successes': [agent0_successes, agent1_successes]
        }
        other_info_dict = {
            'Adversary both targets': [np.average(np.array(adversary_both_targets))],
            'Adversary both targets (v)': [np.var(np.array(adversary_both_targets))],
            'Adversary wrong target': [np.average(np.array(adversary_wrong_target))],
            'Adversary only correct target': [np.average(np.array(agent0_successes))],
            'Agents both close': [np.average(np.array(both_close))],
            'Agents both close (v) ': [np.var(np.array(both_close))],
            'Agents neither close': [np.average(np.array(neither_close))],
            'Agents neither close (v)': [np.var(np.array(neither_close))]
        }
        # if not os.path.exists(prefix):
        #    os.makedirs(prefix)
        df = pd.DataFrame(success_dict)
        df.to_csv(prefix + '.csv')

        ret = str(successes)
        if print_result:
            print(label_info + ': ' + str(successes))
        return ret
    else:
        raise NotImplementedError()


def compute_accuracies_critical_points(xs, ys, experts, students, teachers, threshold=0.05):
    # compute individual accuracies on critical points

    all_qs = [expert.predict_q_for_each_action(x, y) for x, y, expert in zip(xs, ys, teachers)]
    weighted = [np.max(qs, axis=1) - np.min(qs, axis=1) for qs in all_qs]
    critical_pt_indices = [np.where(weight > threshold) for weight in weighted]
    critical_point_states = [x[critical_pt_index] for x, critical_pt_index in zip(xs, critical_pt_indices)]
    critical_point_labels = [y[critical_pt_index] for y, critical_pt_index in zip(ys, critical_pt_indices)]
    critical_pt_accuracies = [accuracy(student, x, y) for x, y, student in
                              zip(critical_point_states, critical_point_labels, students)]
    print('critical pt accuracies are', critical_pt_accuracies)


def compute_accuracies(experts, students, env, n_test_rollouts, max_timesteps, save_folder, save_file_name, teachers,
                       adversaries, good_agents):
    # compare the teacher with the student
    acc_dict = {}
    threshold = 0.1
    combine_qs = False
    xs, ys = [[] for _ in range(experts.nagents)], [[] for _ in range(experts.nagents)]
    crit_xs, crit_ys = [[] for _ in range(experts.nagents)], [[] for _ in range(experts.nagents)]
    critical_state_metric = 'other'
    for rollout_num in range(n_test_rollouts):
        obs, done, num_its = env.reset(), False, 0
        while not done:
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(experts.nagents)]
            torch_actions = experts.step(torch_obs, explore=False)
            all_qs = [teacher.predict_q_for_each_action(torch_obs, torch_actions) for teacher in teachers]

            if critical_state_metric == 'second_best':
                weighted = [np.max(qs, axis=1) - np.partition(qs, -2)[:, -2] for qs in all_qs]
            else:
                weighted = [np.max(qs, axis=1) - np.min(qs, axis=1) for qs in all_qs]
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            assert len(actions) == experts.nagents
            assert len(obs) == experts.nagents
            num_its += 1
            for a in range(experts.nagents):
                xs[a].append(obs[a])
                ys[a].append(actions[a])
                if weighted[a] > threshold:
                    crit_xs[a].append(obs[a])
                    crit_ys[a].append(actions[a])
            obs, rewards, dones, infos = env.step(actions)
            done = num_its >= max_timesteps
            if num_its >= max_timesteps:
                break
    ys = [np.array(y) for y in ys]
    accuracies = [accuracy(student, x, y) for x, y, student in zip(xs, ys, students)]
    critical_accs = [accuracy(student, x, y) for x, y, student in zip(crit_xs, crit_ys, students)]
    print('individual accuracies are', accuracies)
    print('indiv crit accuracies are', critical_accs)
    print('total num of xs are ' + str([len(x) for x in xs]))
    print('total num crits' + str([len(x) for x in crit_xs]))
    # compute_accuracies_critical_points(xs, ys, experts, students, teachers)
    # isolate teams and compute joint accuracies 
    joint_accs = [None for _ in range(len(students))]
    if len(good_agents) > 1:
        total_preds = [
            np.array(ys[i] == students[i].predict(xs[i])) for i in good_agents
        ]
        for total_pred in total_preds:
            print(total_pred.shape)
        sum_preds = [
            np.sum(total_pred, axis=1) for total_pred in total_preds
        ]
        corrects = [
            sum_pred >= total_pred.shape[1] for sum_pred, total_pred in zip(sum_preds, total_preds)
        ]
        joint_correct = np.prod(np.vstack(corrects), axis=0)
        good_total_acc = np.mean(joint_correct)
        for a_idx in good_agents:
            joint_accs[a_idx] = good_total_acc
    elif len(good_agents) == 1:
        joint_accs[good_agents[0]] = accuracies[good_agents[0]]

    # now test joint adversary performance 
    if len(adversaries) > 1:
        total_preds = [
            np.array(ys[i] == students[i].predict(xs[i]) for i in adversaries)
        ]
        sum_preds = [
            np.sum(total_pred, axis=1) for total_pred in total_preds
        ]
        corrects = [
            sum_pred >= total_pred.shape[1] for sum_pred, total_pred in zip(sum_preds, total_preds)
        ]
        joint_correct = np.prod(np.vstack(corrects), axis=0)
        adv_total_acc = np.mean(joint_correct)
        for a_idx in good_agents:
            joint_accs[a_idx] = adv_total_acc
    elif len(adversaries) == 1:
        joint_accs[adversaries[0]] = accuracies[adversaries[0]]

    labels = ['Agent ' + str(i) for i in range(len(students))]
    acc_dict = {
        'Agent': labels,
        'Indiv acc': accuracies,
        'Joint acc': joint_accs
    }
    df = pd.DataFrame(acc_dict)
    df.to_csv(save_folder + save_file_name + '_accuracies.csv')


class WrappedAgents:
    def __init__(self, agents, use_info=False):
        self.agents = agents
        self.use_info = use_info

    def step(self, obs, info=None):
        actions = []
        for i, agent in enumerate(self.agents):
            if self.use_info and info is not None and not (isinstance(agent, MADDPG) or isinstance(agent, DDPGAgent)):
                cur_obs = info[i]
            else:
                cur_obs = obs[i]
            if isinstance(agent, DTPolicy):
                actions.append(agent.predict(np.array([cur_obs]))[0])
            elif isinstance(agent, JointDTPolicy.Individual):
                if len(cur_obs.shape) == 2:
                    actions.append(agent.predict(np.array(cur_obs))[0])
                else:
                    actions.append(agent.predict(np.array([cur_obs]))[0])
            elif isinstance(agent, FittedQDTAgent):
                actions.append(agent.step(np.array(obs[i])))

            elif isinstance(agent, ImitatedDTAgent):
                actions.append(agent.step(np.array(obs[i])))
            else:
                actions.append(agent.step(
                    Variable(torch.Tensor(cur_obs).view(1, -1),
                             requires_grad=False), explore=False).data.numpy().flatten()
                               )
        assert len(actions) == len(self.agents)
        return actions

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, item):
        return self.agents[item]


def estimate_student_quality(students, sa_envs, is_train, nagents, global_seed, state_transformer, n_test_rollouts_eval,
                             scenario_name, selection_criteria, parameters):
    print('estimating student quality...')
    students_with_estimates = [[] for _ in range(len(students))]
    for s_idx, student_set in enumerate(students):
        cur_student_set = student_set
        for indiv_idx, student in enumerate(cur_student_set):
            if is_train:
                student = student[0]
            idx = s_idx
            score_infos, _, _ = test_policy(
                env=sa_envs[idx],
                policy=student,
                state_transformer=state_transformer,
                n_test_rollouts=n_test_rollouts_eval,
                index=idx,
                env_name=scenario_name,
                global_seed=global_seed,
                selection_criteria=selection_criteria
            )
            score_avg, score_var = score_infos
            student_set = (student.clone(), score_avg, score_var, indiv_idx)
            students_with_estimates[s_idx].append(student_set)
    return students_with_estimates


def estimate_joint_student_quality(maddpg, students, env, sa_envs, is_train, nagents, global_seed, state_transformer,
                                   n_test_rollouts_eval,
                                   scenario_name, selection_criteria, parameters, team):
    assert parameters.joint_training
    print('estimating student quality...')
    students_with_estimates = []
    for s_idx, student_set in enumerate(students):
        student = student_set[0]
        total_score_infos = []

        if not parameters.estimate_by_team:
            for indiv_idx, indiv_student in enumerate(student):
                score_infos, _, _ = test_policy(
                    env=sa_envs[indiv_idx],
                    policy=indiv_student,
                    state_transformer=state_transformer,
                    n_test_rollouts=n_test_rollouts_eval,
                    index=indiv_idx,
                    env_name=scenario_name,
                    global_seed=global_seed,
                    selection_criteria=selection_criteria
                )
                score_avg, score_var = score_infos
                total_score_infos.append((score_avg, score_var))
        else:
            # Team up with other students
            total_score_infos = [None] * len(student)
            for t in team.values():
                idx = [i for i in range(len(student)) if team[i] == t]
                agents = [student[a] if a in idx else maddpg.agents[a] for a in range(len(student))]
                score_infos, _, _ = test_policy(
                    env=env,
                    policy=WrappedAgents(agents, use_info=hasattr(env, 'easy_feature_sizes')),
                    state_transformer=state_transformer,
                    n_test_rollouts=n_test_rollouts_eval,
                    index=idx,
                    env_name=scenario_name,
                    global_seed=global_seed,
                    selection_criteria=selection_criteria
                )
                for i, j in enumerate(idx):
                    if isinstance(score_infos[0], list) or isinstance(score_infos[0], np.ndarray):
                        total_score_infos[j] = (score_infos[0][i], score_infos[1][i])
                    else:
                        total_score_infos[j] = (score_infos[0], score_infos[1])

        student_set = (student, [_[0] for _ in total_score_infos], [_[1] for _ in total_score_infos])
        students_with_estimates.append(student_set)
        print(s_idx, student_set[1], sum(student_set[1]))
    return students_with_estimates


def log_rewards(rewards, prefix, print_result=True):
    # rewards should be (num_episodes, num_timesteps, num_agents)
    # calculate the average per-episode accumulated reward per agent
    # calculate the average per-episode reward over all timesteps per agent 
    rewards = np.array(rewards)
    num_agents = rewards.shape[-1]

    # just dump the raw rewards in case we want to parse again later 
    for a_i in range(num_agents):
        np.savetxt(prefix + 'agent' + str(a_i) + '_raw.csv', rewards[:, :, a_i], delimiter=",")

    # how well does each agent do on average over all timesteps in the span of an episode?
    # NOTE: not logging this for now, but can be easily recovered from raw log above
    # NOTE (zhicheng): explicitly logging this
    per_episode_returns = np.sum(rewards, axis=1)  # (num_episodes, num_agents)
    average_per_episode_rewards = np.mean(per_episode_returns, axis=0).T  # (num_agents)
    per_episode_reward_var = np.var(per_episode_returns, axis=0).T

    # how well on average does an agent do for each timestep?
    per_timestep_rewards = np.mean(rewards, axis=0).T  # (num_timesteps, num_agents)
    per_timestep_rewards_var = np.var(rewards, axis=0).T

    # how well do the agents do over the course of an episode?
    cumulative_rewards = np.cumsum(rewards, axis=1)
    average_per_timestep_cumulative_rewards = np.mean(cumulative_rewards, axis=0).T  # (num_timesteps, num_agents)
    per_timestep_cum_rewards_var = np.var(cumulative_rewards, axis=0).T

    reward_dict = {}
    ret = []
    for a_i in range(num_agents):
        label_prefix = 'Agent ' + str(a_i)
        reward_dict[label_prefix + ' per timestep rewards: '] = per_timestep_rewards[a_i]
        reward_dict[label_prefix + ' per timestep rewards var'] = per_timestep_rewards_var[a_i]
        reward_dict[label_prefix + ' average per timestep cum rewards'] = average_per_timestep_cumulative_rewards[a_i]
        reward_dict[label_prefix + ' per timestep cum rewards var'] = per_timestep_cum_rewards_var[a_i]
        reward_dict[label_prefix + ' per episode returns'] = average_per_episode_rewards[a_i]
        reward_dict[label_prefix + ' per episode returns var'] = per_episode_reward_var[a_i]
        ret.append(float('%.2f' % average_per_episode_rewards[a_i]))

    df = pd.DataFrame(reward_dict)
    df.to_csv(prefix + '.csv')

    if print_result:
        print(ret)
    return str(ret)


def log_average(infos, print_result=True, replicate=1):
    infos = np.mean(np.array(infos))
    ret = [float("%.2f" % infos)] * replicate
    if print_result:
        print(ret)
    return ret


def evaluate_agents(agents, env, n_test_rollouts, eval_epsilon, max_timesteps, save_folder,
                    save_file_name, setting, scenario_name, seed_mult=20, team_info=None, test_exploitability=False):
    agents = WrappedAgents(agents, use_info=hasattr(env, 'easy_feature_sizes'))
    infos_to_record, rewards_to_record = [], []
    # This part could be modified to using get_rollout

    for rollout_num in range(n_test_rollouts):
        env._seed(rollout_num * seed_mult)
        obs, done, num_its = env.reset(), False, 0
        ep_info, ep_rew_info = [], []

        # rollout episode 
        while not done:
            info = [env.info_callback(agent, env.world)[1] for agent in env.agents]
            actions = agents.step(obs, info)
            obs, rewards, dones, infos = env.step(actions)
            num_its += 1
            done = num_its >= max_timesteps
            # print('infos are', infos)
            ep_info.append(infos)
            ep_rew_info.append(rewards)
            if num_its >= max_timesteps:
                break
        infos_to_record.append(ep_info)
        rewards_to_record.append(ep_rew_info)

    if scenario_name == 'simple_adversary':
        success_result = log_success_percent(
            infos=infos_to_record,
            prefix=save_folder + save_file_name + '_' + setting,
            epsilon=eval_epsilon,
            label_info=setting,
            scenario_name=scenario_name,
            print_result=False,
        )
    else:
        success_result = 'NaN'

    rewards_result = log_rewards(
        rewards=rewards_to_record,
        prefix=save_folder + save_file_name + '_' + setting + '_rewards',
        print_result=False,
    )
    additional_result = ""

    if scenario_name == 'simple_spread':
        collision_info = [np.sum([step_info['n'][0][1] for step_info in info]) for info in infos_to_record]
        dist_info = [np.mean([step_info['n'][0][2] for step_info in info]) for info in infos_to_record]
        collision_result = log_average(
            infos=collision_info,
            print_result=False,
            replicate=len(env.agents)
        )
        min_dist = log_average(
            infos=dist_info,
            print_result=False,
            replicate=len(env.agents)
        )
        additional_result = f", collision: {collision_result}, min_dist: {min_dist}"
    if scenario_name == 'simple_tag':
        touches = [np.sum([sum(step_info['n'][i][0] for i in range(len(env.agents))) for step_info in info]) for info in infos_to_record]
        touch_result = log_average(
            infos=touches,
            print_result=False,
            replicate=len(env.agents)
        )
        additional_result = f", touches: {touch_result}"

    if test_exploitability:
        # print("team info is", team_info)
        exploitability_result = calc_exploitability(
            agents=agents,
            env=env,
            team_info=team_info,
            save_dir=save_folder + save_file_name + '_' + setting,
            fast=0,
            print_results=False,
        )
        # print('exploitability results are', exploitability_result)
        save_dict = {}
        for a_i in range(len(exploitability_result)):
            save_dict['Agent ' + str(a_i)] = [exploitability_result[a_i]]
        df = pd.DataFrame(save_dict)
        df.to_csv(save_folder + save_file_name + '_' + setting + '_exploitability_team_info.csv')
        print(
            f'{setting}: success rate: {success_result}, reward: {rewards_result}, exploitability: {exploitability_result}{additional_result}')
    else:
        print(f'{setting}: success rate: {success_result}, reward: {rewards_result}{additional_result}')
        return f'{setting}: success rate: {success_result}, reward: {rewards_result}{additional_result}'

def get_env(scenario_name):
    if 'cybersec' in scenario_name:
        env = CybersecScenario()
    else:
        scenario = scenarios.load(scenario_name + '.py').Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            discrete_action=True
        )
    return env


def get_agent_types(env):
    if all([hasattr(a, 'adversary') for a in env.agents]):
        agent_types = ['adversary' if a.adversary else 'agent' for a in
                       env.agents]
    else:
        agent_types = ['agent' for _ in env.agents]
    return agent_types


def setup_save_folder(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    existing_runs = [
        int(
            str(fol).split('run')[1].replace("'", "").replace(">", "")
        ) for fol in os.scandir(save_folder) if fol.is_dir()
    ]
    curr_run = 'run1' if len(existing_runs) == 0 else 'run%i' % (max(existing_runs) + 1)
    save_folder = save_folder + curr_run + '/'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def average_window(array, window_size=50):
    return sum(array[-window_size:]) / len(array[-window_size:])


def get_setup_name(nns, dts, num_total):
    setup = []
    assert len(nns) == 0 or len(dts) == 0 or len(nns) + len(dts) == num_total
    if len(nns) == 0:
        nns = [i for i in range(num_total) if i not in dts]
    elif len(dts) == 0:
        dts = [i for i in range(num_total) if i not in nns]
    for i in range(num_total):
        if i in dts:
            setup.append('DT')
        else:
            setup.append('NN')
    return '_'.join(setup)
