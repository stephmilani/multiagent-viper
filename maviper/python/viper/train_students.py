from core.rl import *
from maviper.python.viper.core.joint_dt import JointDTPolicy
from pong import *
from core.dt import *
from util.log import *
from util.helpers import *
from util.env_wrappers import SingleAgentWrapper
from core.teacher import TeacherFunc
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from multiagent.scenarios.simple_cybersec import CybersecScenario
from algorithms.maddpg import MADDPG
import torch
import wandb
import numpy as np
import time
import os
import argparse
from models.random import RandomAgent
import pandas as pd
import json
import sys

work_dir = os.getcwd()
CSV_RESULTS_FOLDER = f'{work_dir}/maviper/python/viper/results/'


# TODO: might be weird for outcome -- seems acting up?
def log_success_percent(infos, prefix, epsilon, label_info, scenario='simple_adversary'):
    if scenario == 'simple_adversary':
        agent0_successes, agent1_successes, agent0_s, agent1_s = [], [], [], []
        n_tsteps = 25
        for ep in infos:
            agent0_ep_successes, agent1_ep_successes = [], []
            assert len(ep) == n_tsteps
            # data consists of adversary dist, (agent_dists to landmarks, goal)
            for it in ep:
                # check the adversary distance 
                adv_distance_to_t0, adv_distance_to_t1, adv_distance_to_target = it['n'][0]
                # adv_distance_to_target = it['n'][0]
                if adv_distance_to_target < epsilon:
                    agent0_ep_successes.append(int(True))
                else:
                    agent0_ep_successes.append(int(False))

                # if one of the agents is close to the first target
                # and the other agent is close to the other target and vice versa
                if (it['n'][1][0] < epsilon and it['n'][2][1] < epsilon) \
                        or (it['n'][1][1] < epsilon and it['n'][2][0] < epsilon):
                    agent1_ep_successes.append(int(True))
                else:
                    agent1_ep_successes.append(int(False))
            print('Agent succeeded? ', sum(agent0_ep_successes))
            agent0_successes.append(agent0_ep_successes)
            agent0_s.append(int(sum(agent0_ep_successes) > 0))
            agent1_successes.append(agent1_ep_successes)
            agent1_s.append(int(sum(agent1_ep_successes) > 0))
        print('List of succeses', agent0_s)
        labels = ['Adversary', 'Agents']
        successes = [
            sum(agent0_s) / len(agent0_s),
            sum(agent1_s) / len(agent1_s)
        ]
        success_var0 = np.var(np.array(agent0_s))
        success_var1 = np.var(np.array(agent1_s))
        success_dict = {
            'Agent': labels,
            'Success %': successes,
            'Successes': [agent0_successes, agent1_successes],
            'Success Vars': [success_var0, success_var1]

        }
        print(label_info + ': ' + str(successes))
        df = pd.DataFrame(success_dict)
        df.to_csv(prefix + '.csv')


def log_wandb_bar(labels, info, x_label, y_label, loc, title):
    data = [[label, val] for (label, val) in zip(labels, info)]
    table = wandb.Table(data=data, columns=[y_label, x_label])
    wandb.log({loc: wandb.plot.bar(table, y_label, x_label, title=title)})


def learn_dt(parameters):
    # Setting test_gen to True to generate test data
    parameters.test_gen = True

    # Parameters
    max_val = 10000
    global_seed = parameters.random_seed
    scenario_name = parameters.scenario_name
    model_path = f"{work_dir}/models/{scenario_name}/" if len(
        parameters.model_path) == 0 else parameters.model_path
    is_train = parameters.is_train
    incremental = None
    max_depth = parameters.max_depth

    # Logging
    save_file_name = parameters.test_name
    if len(parameters.load_folder) == 0:
        parameters.load_folder = CSV_RESULTS_FOLDER
    if parameters.is_train:
        if parameters.joint_training:
            save_folder = setup_save_folder(
                CSV_RESULTS_FOLDER + scenario_name + '/joint-max-depth' + str(max_depth) + '/'
            )
        else:
            save_folder = setup_save_folder(
                CSV_RESULTS_FOLDER + scenario_name + '/max-depth' + str(max_depth) + '/'
            )
        print('The result will be saved to ' + save_folder)
    else:
        if parameters.joint_training:
            save_folder = parameters.load_folder + scenario_name + '/joint-max-depth' + str(
                max_depth) + '/' + parameters.load_run + '/'
        else:
            save_folder = parameters.load_folder + scenario_name + '/max-depth' + str(
                max_depth) + '/' + parameters.load_run + '/'
        print('The result will be loaded from ' + save_folder)

    if parameters.use_wandb:
        wandb.init(
            name=parameters.test_name + '-max-depth-' + str(parameters.max_depth),
            group=scenario_name,
            project='maviper-debug',
            config=parameters
        )

    # Data structures
    env = get_env(scenario_name)

    # get the proper model path for loading the models 
    model_f = 'model%i.pt' % incremental if incremental is not None else 'model.pt'
    model_path = model_path + model_f

    maddpg = MADDPG.init_from_save(model_path)
    good_agents = [a_idx for a_idx, agent in enumerate(env.world.agents) if not agent.adversary]
    adversaries = [a_idx for a_idx, agent in enumerate(env.world.agents) if agent.adversary]

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

    maddpg.prep_rollouts_q(device='cpu')
    state_transformer = lambda x: x

    # Test MADDPG beforehand
    print("Testing MADDPG")
    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    evaluate_agents(
        agents=maddpg.agents,
        env=env,
        n_test_rollouts=parameters.n_eval_rollouts,
        eval_epsilon=parameters.eval_epsilon,
        max_timesteps=parameters.max_timesteps,
        save_folder=save_folder,
        save_file_name=save_file_name,
        setting='experts',
        seed_mult=global_seed,
        scenario_name=scenario_name,
        team_info=team_info,
        test_exploitability=False,
    )
    print('Logged MADDPG evaluation')

    # set up the students, teachers, and associated single-agent environments
    students = [DTPolicy(parameters.max_depth) for _ in range(maddpg.nagents)]
    sa_envs = [SingleAgentWrapper(env, maddpg.agents, t_i, False, scenario_name) for t_i in range(maddpg.nagents)]
    teachers = [TeacherFunc(sa_envs[i], maddpg.agents[i], i, team=team_info) for i in range(maddpg.nagents)]
    student_data_infos = [[] for _ in range(maddpg.nagents)]

    # train the students or pull them from stored folder
    if is_train:
        with open(str(save_folder) + 'args.txt', 'w') as f:
            json.dump(parameters.__dict__, f, indent=2)
        start_train_time = time.time()
        if parameters.joint_training:
            student = JointDTPolicy(max_depth=parameters.max_depth, env=env, num_policies=maddpg.nagents,
                                    team_info=team_info)
            teacher = maddpg
            students, _ = train_dagger_joint(
                env=env,
                teacher=teacher,
                student=student,
                state_transformer=state_transformer,
                max_iters=parameters.max_iters,
                train_frac=parameters.train_frac,
                n_batch_rollouts=parameters.n_batch_rollouts,
                max_samples=parameters.max_samples,
                is_reweight=parameters.is_reweight,
                n_test_rollouts=parameters.n_test_rollouts,
                env_name=parameters.scenario_name,
                save_folder=save_folder + 'models/',
                save_file_name=save_file_name,
                parameters=parameters,
                selection_criteria=parameters.selection_metric
            )
        else:
            for teacher_index in range(maddpg.nagents):
                student, student_data = train_dagger(
                    teachers=teachers,
                    student=students[teacher_index],
                    env=sa_envs[teacher_index],
                    state_transformer=state_transformer,
                    max_samples=parameters.max_samples,
                    max_iters=parameters.max_iters,
                    train_frac=parameters.train_frac,
                    n_batch_rollouts=parameters.n_batch_rollouts,
                    is_reweight=parameters.is_reweight,
                    n_test_rollouts=parameters.n_test_rollouts,
                    teacher_index=teacher_index,
                    env_name=parameters.scenario_name,
                    save_folder=save_folder + 'models/',
                    save_file_name=save_file_name,
                    parameters=parameters,
                    selection_criteria=parameters.selection_metric
                )
                students[teacher_index] = student
                student_data_infos[teacher_index] = student_data
        train_students_time = time.time()
    else:
        start_train_time = time.time()
        if parameters.scenario_name == 'simple_tag' and parameters.easy_feature:
            env.easy_feature_sizes = True

        if parameters.joint_training:
            students = []
            student_fol = save_folder + '/models/agent/'
            for filename in os.listdir(student_fol):
                students.append(load_dt_policy(student_fol, filename))
        else:
            students = [[] for _ in range(maddpg.nagents)]
            for s in range(maddpg.nagents):
                student_fol = save_folder + '/models/agent' + str(s) + '/'
                for filename in os.listdir(student_fol):
                    with open(os.path.join(student_fol, filename), 'r') as f:
                        stud = load_dt_policy(
                            student_fol, filename
                        )
                        students[s].append(stud)
        train_students_time = time.time()
    student_train_time = train_students_time - start_train_time

    # Redirect stdout to file so that we can view the results later
    if not parameters.print_to_console:
        sys.stdout = open(
            save_folder + '/exploit_eval_results' + (
                f'_{parameters.test_only}' if parameters.test_only != -1 else '') + (
                '_indep' if parameters.independent_policy_selection else '') + (
                f'_{parameters.selection_metric}') + ('_team' if parameters.estimate_by_team else '') + '.txt',
            'w')

    # now that we have the students, they need to have an initial estimate of their quality
    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    if not parameters.joint_training:
        students = estimate_student_quality(
            students=students,
            sa_envs=sa_envs,
            is_train=is_train,
            nagents=maddpg.nagents,
            global_seed=global_seed,
            state_transformer=state_transformer,
            n_test_rollouts_eval=parameters.n_test_rollouts_eval,
            scenario_name=scenario_name,
            selection_criteria=parameters.selection_metric,
            parameters=parameters,
        )
    else:
        students = estimate_joint_student_quality(
            maddpg=maddpg,
            students=students,
            env=env,
            sa_envs=sa_envs,
            is_train=is_train,
            nagents=maddpg.nagents,
            global_seed=global_seed,
            state_transformer=state_transformer,
            n_test_rollouts_eval=parameters.n_test_rollouts_eval,
            scenario_name=scenario_name,
            selection_criteria=parameters.selection_metric,
            parameters=parameters,
            team=team_info,
        )
    student_quality_time = time.time() - train_students_time

    best_students, best_student_infos = [None for _ in range(maddpg.nagents)], [None for _ in range(maddpg.nagents)]
    student_selection_times = [None for _ in range(maddpg.nagents)]
    # get the best students from the set
    if len(students[0]) > 0:
        if parameters.joint_training:
            best_students, best_student_infos = identify_best_joint_policy(
                policies=students,
                parameters=parameters,
                team=team_info,
            )
        elif parameters.independent_policy_selection:
            for i in range(maddpg.nagents):
                global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
                best_student = identify_best_policy(
                    env=sa_envs[i],
                    policies=students[i],
                    state_transformer=state_transformer,
                    n_test_rollouts=parameters.n_test_rollouts_eval,
                    teacher_index=i,
                    global_seed=global_seed,
                    env_name=scenario_name
                )
                pol, _, _, idx = best_student
                best_students[i] = pol
                best_student_infos[i] = student_data_infos[i][idx]
                student_selection_times[i] = time.time()
        else:
            eval_against_expert = True  # whether to pick the best policies when pitted against the expert or DTs. atm, doesnt matter for adversaries given sequence.
            wrapped_agents = [TeacherFunc(sa_env, agent, a, team=team_info) for a, (sa_env, agent) in
                              enumerate(zip(sa_envs, maddpg.agents))]
            agent_policies = [a_pol if a in adversaries else wrapped_agents[a] for a, a_pol in enumerate(students)]
            # first, we get the best adversaries
            print("Adversary is", len(adversaries))
            if len(adversaries) > 0:
                if len(adversaries) > 1:
                    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
                    best_adversary = identify_best_joint_policies(
                        env=env,
                        policies=agent_policies,
                        n_test_rollouts=parameters.n_test_rollouts_eval,
                        n_init_pol=parameters.n_init_pol,
                        team_indices=adversaries,
                        global_seed=global_seed,
                        selection_criteria=parameters.selection_metric,
                        env_name=scenario_name
                    )
                else:
                    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
                    best_adversary = identify_best_policy(
                        sa_envs[adversaries[0]],
                        students[adversaries[0]],
                        state_transformer,
                        parameters.n_test_rollouts_eval,
                        adversaries[0],
                        global_seed=global_seed
                    )
                    best_adversary = [best_adversary]
                    print('best adversary is ', best_adversary)

                selection_time = time.time()
                # iterate through and update with dt adversaries
                for s in range(len(best_adversary)):
                    pol, _, _, idx = best_adversary[s]
                    best_students[adversaries[s]] = pol
                    best_student_infos[adversaries[s]] = student_data_infos[adversaries[s]][idx]
                    student_selection_times[adversaries[s]] = selection_time
                print('Done choosing adversary')  # TODO: update to include adversary selection time

            if len(good_agents) > 0:
                if eval_against_expert:
                    agent_policies = [a_pol if a not in adversaries else wrapped_agents[a] for a, a_pol in
                                      enumerate(students)]
                else:
                    agent_policies = [a_pol for a, a_pol in enumerate(students)]

                if len(good_agents) > 1:
                    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
                    best_team = identify_best_joint_policies(
                        env=env,
                        policies=agent_policies,
                        state_transformer=state_transformer,
                        n_test_rollouts=parameters.n_test_rollouts_eval,
                        n_init_pol=parameters.n_init_pol,
                        global_seed=global_seed,
                        selection_criteria=parameters.selection_metric,
                        team_indices=good_agents,
                        env_name=scenario_name
                    )
                    print('best team is', best_team)
                else:
                    pass  # so far no envs with only one defender. TODO: implement
                agent_selection_time = time.time()

                for s in range(len(best_team)):
                    pol, _, _, idx = best_team[s]
                    best_students[good_agents[s]] = pol
                    best_student_infos[good_agents[s]] = student_data_infos[s][
                        idx]  # TODO: not sure if correctly updating this
                    student_selection_times[good_agents[s]] = agent_selection_time

        print('Done identifying policies')
        students = best_students
        # TODO: implement getting times correctly
        times_dict = {}
        for s, student_time in enumerate(student_selection_times):
            times_dict['Student' + str(s) + ' Raw Selection Time '] = [student_time]
        times_dict['Student Quality Evaluation Time'] = [student_quality_time]
        times_dict['Student Train Time'] = [student_train_time]
        times_dict['Student Raw Start Train Time'] = [start_train_time]

        pd.DataFrame(times_dict).to_csv(save_folder + save_file_name + '_times.csv')
        for s, student_data_dict in enumerate(best_student_infos):
            print('student info' + str(s), student_data_dict)
            pd.DataFrame(student_data_dict).to_csv(save_folder + save_file_name + '_agent' + str(s) + '.csv')

    # Save the final policy
    if parameters.is_train:
        print('Saved to', save_folder)
        with open(save_folder + '/' + 'final_policy.pk', 'wb') as f:
            pickle.dump(students, f)
    else:
        # Try loading the policy
        with open(save_folder + ('/' if save_folder[-1] != '/' else '') + 'final_policy.pk', 'rb') as f:
            students = pickle.load(f)
            try:
                if len(students) > 0 \
                        and len(students[0]) > 0 \
                        and (isinstance(students[0][0], JointDTPolicy.Individual)
                             or isinstance(students[0][0], JointDTPolicy)
                             or isinstance(students[0][0], DTPolicy)):
                    students = students[0]
            except:
                pass

    if parameters.visualize:
        if hasattr(students[0], 'visualize'):
            for i, student in enumerate(students):
                student.visualize(f'{save_folder}student_visualization_{i}')
        else:
            raise NotImplementedError('Visualization not implemented for this kind of student')

    # now we move to evaluation...
    print('Starting evaluation...\n')
    if parameters.compute_accuracies:
        compute_accuracies(
            experts=maddpg,
            students=students,
            env=env,
            n_test_rollouts=parameters.n_test_rollouts,
            max_timesteps=parameters.max_timesteps,
            save_folder=save_folder,
            save_file_name=save_file_name,
            teachers=teachers,
            adversaries=adversaries,
            good_agents=good_agents
        )

    # test the maddpg agents against each other
    print("Testing MADDPG Again")
    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    evaluate_agents(
        agents=maddpg.agents,
        env=env,
        n_test_rollouts=parameters.n_eval_rollouts,
        eval_epsilon=parameters.eval_epsilon,
        max_timesteps=parameters.max_timesteps,
        save_folder=save_folder,
        save_file_name=save_file_name,
        setting='experts',
        seed_mult=global_seed,
        scenario_name=scenario_name,
        test_exploitability=parameters.evaluate_exploitability,
        team_info=team_info,
    )
    print('Logged MADDPG evaluation')

    # for each index, test nn agent at that index vs dt agents and dt agent vs nn agents
    for nn_index in range(len(maddpg.agents)):
        print("Evaluating Agent " + str(nn_index) + " against DTs, NNs")
        global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)

        evaluate_agents(
            agents=[maddpg.agents[nn_index] if a == nn_index else students[a] for a in range(len(maddpg.agents))],
            env=env,
            n_test_rollouts=parameters.n_eval_rollouts,
            eval_epsilon=parameters.eval_epsilon,
            max_timesteps=parameters.max_timesteps,
            save_folder=save_folder,
            save_file_name=save_file_name,
            setting=get_setup_name([nn_index], [], num_total=len(maddpg.agents)),
            seed_mult=global_seed,
            scenario_name=scenario_name,
            test_exploitability=parameters.evaluate_exploitability,
            team_info=team_info,
        )
        global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
        evaluate_agents(
            agents=[students[nn_index] if a == nn_index else maddpg.agents[a] for a in range(len(maddpg.agents))],
            env=env,
            n_test_rollouts=parameters.n_eval_rollouts,
            eval_epsilon=parameters.eval_epsilon,
            max_timesteps=parameters.max_timesteps,
            save_folder=save_folder,
            save_file_name=save_file_name,
            setting=get_setup_name([], [nn_index], num_total=len(maddpg.agents)),
            seed_mult=global_seed,
            scenario_name=scenario_name,
            test_exploitability=parameters.evaluate_exploitability,
            team_info=team_info,
        )
    print("Logged individual agent evaluation")

    # Evaluate team
    if parameters.scenario_name != 'simple_adversary':
        for t in set(team_info.values()):
            global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
            evaluate_agents(
                agents=[maddpg.agents[a] if team_info[a] == t else students[a] for a in range(len(maddpg.agents))],
                env=env,
                n_test_rollouts=parameters.n_eval_rollouts,
                eval_epsilon=parameters.eval_epsilon,
                max_timesteps=parameters.max_timesteps,
                save_folder=save_folder,
                save_file_name=save_file_name,
                setting=get_setup_name([a for a in range(len(maddpg.agents)) if team_info[a] == t], [],
                                       num_total=len(maddpg.agents)),
                seed_mult=global_seed,
                scenario_name=scenario_name,
                test_exploitability=parameters.evaluate_exploitability,
                team_info=team_info,
            )

    print('Evaluating all students')
    global_seed = global_seed if not parameters.test_gen else np.random.randint(max_val)
    evaluate_agents(
        agents=students,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', default=4, type=int)
    parser.add_argument('--n_batch_rollouts', default=10, type=int)
    parser.add_argument('--max_samples', default=300000, type=int)
    parser.add_argument('--max_iters', default=30, type=int)
    parser.add_argument('--train_frac', default=0.8, type=float)
    parser.add_argument('--is_reweight', default=True, type=bool)
    parser.add_argument('--n_test_rollouts', default=10, type=int)
    parser.add_argument('--eval_epsilon', default=0.1, type=float)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--scenario_name', default='simple_adversary', type=str)
    parser.add_argument('--max_timesteps', default=25, type=int)
    parser.add_argument('--save_gifs', action='store_true')
    parser.add_argument('--fps', default=5, type=int)
    parser.add_argument('--test_name', default='iviper-jps', type=str)
    parser.add_argument('--turn_based', action='store_true')
    parser.add_argument('--num_inner_loops', default=5, type=int)  # 20
    parser.add_argument('--compute_accuracies', action='store_true')
    parser.add_argument('--independent_policy_selection', action='store_true')
    parser.add_argument('--n_init_pol', default=10, type=int)
    parser.add_argument('--n_eval_rollouts', default=100, type=int)
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--load_folder', type=str, default='')
    parser.add_argument('--load_run', type=str, default='run1')
    parser.add_argument('--random_seed', type=int, default=666)
    parser.add_argument('--n_batch_rollouts_eval', type=int, default=100)
    parser.add_argument('--n_test_rollouts_eval', type=int, default=100)
    parser.add_argument('--test_gen', action='store_true')
    parser.add_argument('--selection_metric', default='reward', type=str)
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--joint_training', action='store_true')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--estimate_by_team', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--easy_feature', action='store_true', default=True)
    parser.add_argument('--evaluate_exploitability', action='store_true')
    parser.add_argument('--optimal_others', action='store_true')
    parser.add_argument('--maxmin', action='store_true')
    parser.add_argument('--average_others', action='store_true')
    parser.add_argument('--test_only', type=int, default=-1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--print_to_console', action='store_true')
    parser.add_argument('--not_joint', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_args()
    learn_dt(parameters)
