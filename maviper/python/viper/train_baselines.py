import argparse
import os
import sys

import torch
import numpy as np
import pandas as pd
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from algorithms.maddpg import MADDPG
from utils.agents import FittedQDTAgent, ImitatedDTAgent
from util.helpers import *
from torch.autograd import Variable
import time
import json
import pandas as pd

work_dir = os.getcwd()
CSV_RESULTS_FOLDER = f'{work_dir}/maviper/python/viper/results/'


# log success %\
def log_success_percent(infos, prefix, epsilon, label_info, scenario='simple_adversary'):
    if scenario == 'simple_adversary':
        agent0_successes, agent1_successes, agent0_s, agent1_s = [], [], [], []
        n_tsteps = 25
        for ep in infos:
            agent0_ep_successes = []
            # print('len ep is', len(ep))
            assert len(ep) == n_tsteps
            # data consists of adversary dist, (agent_dists to landmarks, goal)
            for it in ep:
                if it['n'][0] < epsilon:
                    # close_enough = True
                    agent0_ep_successes.append(int(True))
                    # break
                else:
                    agent0_ep_successes.append(int(False))
            agent0_successes.append(agent0_ep_successes)
            agent0_s.append(int(sum(agent0_ep_successes) > 0))

        # if not close_enough:
        #    agent0_successes.append(int(False))

        for ep in infos:
            # close_enough, both_close, both_covered = False, False, False
            agent1_ep_successes = []
            for it in ep:
                # if one of the agents is close to the first target
                # and the other agent is close to the other target and vice versa
                if (it['n'][1][0] < epsilon and it['n'][2][1] < epsilon) \
                        or (it['n'][1][1] < epsilon and it['n'][2][0] < epsilon):
                    agent1_ep_successes.append(int(True))
                else:
                    agent1_ep_successes.append(int(False))
            agent1_successes.append(agent1_ep_successes)
            agent1_s.append(int(sum(agent1_ep_successes) > 0))

        labels = ['Adversary', 'Agents']
        successes = [
            sum(agent0_s) / len(agent0_s),
            sum(agent1_s) / len(agent1_s)
        ]
        success_dict = {
            'Agent': labels,
            'Success %': successes,
            'Successes': [agent0_successes, agent1_successes]
        }
        df = pd.DataFrame(success_dict)
        print(df)
        # df.to_csv(prefix + '.csv')


def collect_dataset(env, n_agents, max_timesteps, collection_method, max_samples, maddpg=None):
    dataset = [[] for _ in range(n_agents)]

    obs, done = env.reset(), False
    n_iter = 0
    for i in range(max_samples):
        if collection_method == 'random':
            a_acts = [np.zeros(env.action_space[a].n) for a in range(n_agents)]
            agent_acts = [np.random.choice(np.arange(env.action_space[a].n)) for a in range(n_agents)]
            actions = []
            for a_act, ag_act in zip(a_acts, agent_acts):
                a_act[ag_act] = 1.
                actions.append(np.array(a_act))
        elif collection_method == 'maddpg':
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            torch_actions = maddpg.step(torch_obs, explore=False)
            actions = [ac.data.numpy().flatten() for ac in torch_actions]

        nobs, rewards, dones, infos = env.step(actions)

        n_iter += 1
        # TODO: question of what is reasonable to store
        # let's just do simple case for now -- obs and own agent act 
        for a in range(n_agents):
            dataset[a].append((obs[a], actions[a], rewards[a], nobs[a]))

        obs = env.reset() if n_iter >= max_timesteps else nobs
    return dataset


def train_dts(parameters):
    # parameters
    max_val = 10000
    scenario_name = parameters.scenario_name
    max_depth = parameters.max_depth
    global_seed = parameters.random_seed

    # logging 
    save_file_name = parameters.test_name
    # save_folder = setup_save_folder(
    save_folder = CSV_RESULTS_FOLDER + scenario_name + f'/{parameters.agent_type}-max-depth' + str(
        max_depth) + '/run' + parameters.load_run
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # print(save_folder)

    sys.stdout = open(save_folder + '/exploit_eval_results.txt', 'w')

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

    if parameters.collection_method == 'maddpg':
        model_info = parameters.model_path
        model_f = 'model.pt'
        model_path = model_info + model_f
        maddpg = MADDPG.init_from_save(model_path)
        maddpg.prep_rollouts(device='cpu')
    else:
        maddpg = None
    if scenario_name == 'simple_adversary' or scenario_name == 'simple_spread':
        n_agents = 3
    if scenario_name == 'simple_tag':
        n_agents = 4

    if parameters.scenario_name == 'simple_adversary':
        team_info = dict()
        for agent in range(maddpg.nagents):
            team_info[agent] = 0 if env.agents[agent].adversary else 1
    elif parameters.scenario_name == 'simple_spread':
        team_info = dict()
        for agent in range(maddpg.nagents):
            team_info[agent] = 0
    elif parameters.scenario_name == 'simple_tag':
        team_info = dict()
        for agent in range(maddpg.nagents):
            team_info[agent] = 0 if env.agents[agent].adversary else 1
    else:
        raise ValueError('Unknown scenario name')
    #
    # # collect dataset for training
    # dataset_collection_start = time.time()
    # dataset = collect_dataset(
    #     env=env,
    #     n_agents=n_agents,
    #     max_timesteps=parameters.max_timesteps,
    #     collection_method=parameters.collection_method,
    #     max_samples=parameters.max_samples,
    #     maddpg=maddpg
    # )
    # dataset_collection_end = time.time()
    # print("Collected dataset!")
    #
    # # set up agents
    # if parameters.agent_type == 'fittedq':
    #     agents = [
    #         FittedQDTAgent(
    #             training_data=dataset[a],
    #             max_depth=parameters.max_depth,
    #             max_iters=parameters.max_iters,
    #             init_q_val=0.0,
    #             n_acts=env.action_space[a].n
    #         ) for a in range(n_agents)
    #     ]
    # else:
    #     agents = [
    #         ImitatedDTAgent(
    #             training_data=dataset[a],
    #             max_depth=parameters.max_depth,
    #             max_iters=parameters.max_iters,
    #             n_acts=env.action_space[a].n
    #         ) for a in range(n_agents)
    #     ]
    #
    # agent_start_train_times = []
    # agent_end_train_times = []
    # for a, agent in enumerate(agents):
    #     print('Training agent ' + str(a))
    #     agent_start_train_times.append(time.time())
    #     agent.train()
    #     agent_end_train_times.append(time.time())
    # with open(save_folder + '/final_policy.pk', 'wb') as f:
    #     pickle.dump(agents, f)
    # print("Finished training!")

    with open(save_folder + '/' + 'final_policy.pk', 'rb') as f:
        agents = pickle.load(f)

    # times_dict = {
    #     'Dataset collection time': [dataset_collection_end - dataset_collection_start]
    # }
    # for a in range(len(agents)):
    #     times_dict['Agent ' + str(a) + 'train time'] = [agent_end_train_times[a] - agent_start_train_times[a]]
    # pd.DataFrame(times_dict).to_csv(save_folder + 'times.csv')

    # print("Testing MADDPG Again")
    # global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    # evaluate_agents(
    #     agents=maddpg.agents,
    #     env=env,
    #     n_test_rollouts=parameters.n_eval_rollouts,
    #     eval_epsilon=parameters.eval_epsilon,
    #     max_timesteps=parameters.max_timesteps,
    #     save_folder=save_folder,
    #     save_file_name=save_file_name,
    #     setting='experts',
    #     seed_mult=global_seed,
    #     scenario_name=scenario_name,
    #     test_exploitability=parameters.evaluate_exploitability,
    #     team_info=team_info,
    # )
    # print('Logged MADDPG evaluation')
    #
    # for nn_index in range(len(agents)):
    #     print('Evaluating Agent ' + str(nn_index) + ' against DTs, NNs')
    #     global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    #     evaluate_agents(
    #         agents=[agents[nn_index] if a == nn_index else maddpg.agents[a] for a in range(len(maddpg.agents))],
    #         env=env,
    #         n_test_rollouts=parameters.n_eval_rollouts,
    #         eval_epsilon=parameters.eval_epsilon,
    #         max_timesteps=parameters.max_timesteps,
    #         save_folder=save_folder,
    #         save_file_name=save_file_name,
    #         setting=get_setup_name([], [nn_index], num_total=len(maddpg.agents)),
    #         # setting='agent_' + str(nn_index) + 'dt_others_nns',
    #         seed_mult=global_seed,
    #         scenario_name=scenario_name,
    #         test_exploitability=parameters.evaluate_exploitability,
    #         team_info=team_info,
    #     )
    #
    #     global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    #     evaluate_agents(
    #         agents=[maddpg.agents[nn_index] if a == nn_index else agents[a] for a in range(len(maddpg.agents))],
    #         env=env,
    #         n_test_rollouts=parameters.n_eval_rollouts,
    #         eval_epsilon=parameters.eval_epsilon,
    #         max_timesteps=parameters.max_timesteps,
    #         save_folder=save_folder,
    #         save_file_name=save_file_name,
    #         setting=get_setup_name([nn_index], [], num_total=len(maddpg.agents)),
    #         # setting= 'agent_' + str(nn_index) + 'nn_others_dts',
    #         seed_mult=global_seed,
    #         scenario_name=scenario_name,
    #         test_exploitability=parameters.evaluate_exploitability,
    #         team_info=team_info,
    #     )
    # print("Logged individual agent evaluation")
    #
    # # Evaluate team
    # if parameters.scenario_name != 'simple_adversary':
    #     for t in set(team_info.values()):
    #         global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    #         evaluate_agents(
    #             agents=[maddpg.agents[a] if team_info[a] == t else agents[a] for a in range(len(maddpg.agents))],
    #             env=env,
    #             n_test_rollouts=parameters.n_eval_rollouts,
    #             eval_epsilon=parameters.eval_epsilon,
    #             max_timesteps=parameters.max_timesteps,
    #             save_folder=save_folder,
    #             save_file_name=save_file_name,
    #             setting=get_setup_name([a for a in range(len(maddpg.agents)) if team_info[a] == t], [],
    #                                    num_total=len(maddpg.agents)),
    #             # setting= 'agent_' + str(nn_index) + 'nn_others_dts',
    #             seed_mult=global_seed,
    #             scenario_name=scenario_name,
    #             test_exploitability=parameters.evaluate_exploitability,
    #             team_info=team_info,
    #         )

    print('Evaluating all students')
    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    evaluate_agents(
        agents=agents,
        env=env,
        n_test_rollouts=parameters.n_eval_rollouts,
        eval_epsilon=parameters.eval_epsilon,
        max_timesteps=parameters.max_timesteps,
        save_folder=save_folder,
        save_file_name=save_file_name,
        setting='all_dts',
        seed_mult=global_seed,
        scenario_name=scenario_name,
        test_exploitability=parameters.evaluate_exploitability,
        team_info=team_info,
    )
    with open(str(save_folder) + 'args.txt', 'w') as f:
        json.dump(parameters.__dict__, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', default=6, type=int)
    parser.add_argument('--n_batch_rollouts', default=10, type=int)
    parser.add_argument('--max_samples', default=30000, type=int)
    parser.add_argument('--max_iters', default=5, type=int)
    parser.add_argument('--train_frac', default=0.8, type=float)
    parser.add_argument('--is_reweight', default=True, type=bool)
    parser.add_argument('--n_test_rollouts', default=1000, type=int)
    parser.add_argument('--eval_epsilon', default=0.1, type=float)
    parser.add_argument('--collection_method', default='maddpg', type=str)
    parser.add_argument('--load_run', type=str)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--scenario_name', default='simple_adversary', type=str)
    parser.add_argument('--max_timesteps', default=25, type=int)
    parser.add_argument('--save_gifs', action='store_true')
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument('--test_name', default='saviper-indep-pol', type=str)
    parser.add_argument('--n_eval_rollouts', default=100, type=int)
    parser.add_argument('--test_gen', action='store_true')
    parser.add_argument('--random_seed', type=int, default=666)
    parser.add_argument('--evaluate_exploitability', action='store_true')
    parser.add_argument('--agent_type', type=str, default='imitate')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_args()
    train_dts(parameters)
