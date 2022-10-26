import argparse
import os
import pickle
import re

import numpy as np

from maviper.python.viper.util.helpers import get_env, evaluate_agents

result = dict()
work_dir = os.getcwd()


def compare(parameters):
    scenario_name = parameters.scenario_name
    agent_type1 = parameters.agent_type1
    agent_type2 = parameters.agent_type2
    save_folder = f'{work_dir}/maviper/python/viper/results/head_to_head/{scenario_name}/{agent_type1}-{agent_type2}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    env = get_env(scenario_name)
    if parameters.scenario_name == 'simple_adversary':
        team_info = dict()
        for agent in range(3):
            team_info[agent] = 0 if env.agents[agent].adversary else 1
    elif parameters.scenario_name == 'simple_tag':
        team_info = dict()
        for agent in range(4):
            team_info[agent] = 0 if env.agents[agent].adversary else 1
    else:
        raise ValueError('Unknown scenario name')

    # Read agent
    with open(parameters.agent1_location, 'rb') as f:
        agent1 = pickle.load(f)
        # if agent_type1 == 'maviper':
        #     # measure feature importance
        #     importances = []
        #     for agent in agent1:
        #         importances.append(agent.measure_feature_importance())
        #         print(agent.measure_feature_importance())
        #     # os.remove(parameters.agent1_location[:-len('final_policy.pk')] + 'feature_importance.csv')
        #     with open(parameters.agent1_location[:-len('final_policy.pk')] + 'feature_importance.pk', 'wb') as f2:
        #         pickle.dump(importances, f2)
        #     return

    with open(parameters.agent2_location, 'rb') as f:
        agent2 = pickle.load(f)

    global_seed = np.random.randint(1, 100000)

    for t in set(team_info.values()):
        setting = ''.join(['1' if team_info[a] == t else '2' for a in range(len(agent1))])
        ret = evaluate_agents(
            agents=[agent1[a] if team_info[a] == t else agent2[a] for a in range(len(agent1))],
            env=env,
            n_test_rollouts=parameters.n_eval_rollouts,
            eval_epsilon=0.1,
            max_timesteps=25,
            save_folder=save_folder,
            save_file_name=parameters.agent_type1 + '-' + parameters.agent_type2 + '-team' + str(t),
            setting=setting,
            seed_mult=global_seed,
            scenario_name=scenario_name,
            test_exploitability=False,
            team_info=team_info,
        )
        cur_results = []
        for i in range(len(re.split('[\[\]]', ret))):
            try:
                x = re.split('[\[\]]', ret)[i]
                cur_results.append(list(eval(x)))
            except:
                pass
        if parameters.scenario_name == 'simple_tag':
            cur_results[0] = [np.mean(cur_results[0][0:2]), np.mean(cur_results[0][2:])]
            cur_results[1] = [cur_results[1][0]]
        if setting not in result:
            result[setting] = [cur_results]
        else:
            result[setting].append(cur_results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', type=str)
    parser.add_argument('--n_eval_rollouts', default=100, type=int)
    parser.add_argument('--random_seed', type=int, default=666)
    parser.add_argument('--agent_type1', type=str)
    parser.add_argument('--agent1_start', type=str)
    parser.add_argument('--agent_type2', type=str)
    parser.add_argument('--agent2_start', type=str)
    parser.add_argument('--depth', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_args()
    type_dict = {
        'maviper': 'joint-max-depth',
        'iviper': 'max-depth',
        # 'fittedq': 'fittedq-max-depth',
        'fittedq': 'max-depth',
        'imitate': 'imitate-max-depth',
    }

    for i in range(10):
        agent1_id = int(parameters.agent1_start) + i
        agent2_id = int(parameters.agent2_start) + i
        parameters.agent1_location = f'{work_dir}/maviper/python/viper/results/{parameters.scenario_name}/{type_dict[parameters.agent_type1]}{parameters.depth}/run{agent1_id}/final_policy.pk'
        parameters.agent2_location = f'{work_dir}/maviper/python/viper/results/{parameters.scenario_name}/{type_dict[parameters.agent_type2]}{parameters.depth}/run{agent2_id}/final_policy.pk'
        if parameters.agent_type1 == 'fittedq':
            parameters.agent1_location = parameters.agent1_location.replace('MAVIPER',
                                                                            'MAVIPER/maviper_baselines/MAVIPER').replace(
                'results', 'results/04_03_baselines')
        if parameters.agent_type2 == 'fittedq':
            parameters.agent2_location = parameters.agent2_location.replace('MAVIPER',
                                                                            'MAVIPER/maviper_baselines/MAVIPER').replace(
                'results', 'results/04_03_baselines')
        compare(parameters)
    print('Summary of results:')
    for setup, metrics in result.items():
        # print(setup)
        res = ""
        for i in range(len(metrics[0])):
            all_data = [metrics[j][i] for j in range(len(metrics))]
            all_data = np.array(all_data)
            mean = list(np.mean(all_data, axis=0))
            mean = [str(round(x, 2)) for x in mean]
            std = list(np.std(all_data, axis=0))
            std = [str(round(x, 2)) for x in std]
            res += f'{" ".join(mean)} {" ".join(std)} '
        print(res)
