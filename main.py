import argparse
import sys

import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from maviper.python.viper.util.helpers import calc_success_info, average_window
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from tqdm import tqdm
import json

USE_CUDA = torch.cuda.is_available()


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    fully_cooperative = config.env_id == 'simple_spread'

    model_dir = Path(str(os.getcwd()) + '/models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    print('Log at:', str(log_dir))
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.n_training_threads)
    device = 'cuda' if USE_CUDA else 'cpu'

    if 'cybersec' in config.env_id:
        graph = {
            0: [[1, 0.5, 0, 1], [2, 0.5, 1, 2], [3, 0.5, 2, 1]],
            1: [[3, 0.5, 3, 1]],
            2: [],
            3: [[2, 0.5, 4, 1]]
        }
        env = make_env(
            config.env_id, discrete_action=config.discrete_action, graph=graph)
        env.seed(config.seed)
    else:
        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                                config.discrete_action)

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    # sys.stdout = open(str(run_dir / 'easy_summary.txt'), 'w')

    t = 0
    adversary_successes, agent_successes = [], []
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))
        obs = env.reset()
        # print('obs is ', obs, maddpg.nagents)
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device=device)

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        obs = np.array(obs)  # changed from np.array([obs]) for other env
        done = False
        ts = 0
        # adversary_success, agent_success = False, False
        episode_infos = []

        while not done:
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.tensor(np.vstack(obs[:, i]), device=device, dtype=torch.float),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays

            if device == 'cuda':
                agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]
            else:
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # print('agent acts', agent_actions)
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # print('actions are', actions)
            next_obs, rewards, dones, infos = env.step(actions)
            episode_infos.append(infos)

            # adversary_success = adversary_success or calc_success_info(infos[0], scenario=config.env_id)[0]
            # agent_success = agent_success or calc_success_info(infos[0], scenario=config.env_id)[1]

            ts += 1
            # print('dones are', dones)
            # print('dones are', type(dones))
            done = bool(sum([d for d in dones[0]]))
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones, cur_ep=ep_i)
            obs = next_obs
            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                maddpg.prep_training(device=device)
                for u_i in range(config.n_rollout_threads):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                    for a_i in range(maddpg.nagents):
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device=device)
            if ts >= config.episode_length:  # change this if needed for other environments
                break

        # # Calc and log success rate during training
        # adversary_successes.append(adversary_success)
        # agent_successes.append(agent_success)
        # adversary_success, agent_success = average_window(adversary_successes), average_window(agent_successes)
        # logger.add_scalars('success', {'adversary': adversary_success, 'agent_successes': agent_success},
        #                    global_step=ep_i)

        ep_rews = replay_buffer.get_total_rewards(
            config.episode_length * config.n_rollout_threads)
        if ep_i % config.save_interval == 0:
            # print(f'Episode {ep_i}', {'adversary': adversary_success, 'agent': agent_success})
            if fully_cooperative:
                n_test_episode = 10
                total_reward = 0
                for _ in range(n_test_episode):
                    obs, done, t = env.reset(), False, 0
                    while not done:
                        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                              requires_grad=False).to(device)
                                     for i in range(maddpg.nagents)]
                        torch_agent_actions = maddpg.step(torch_obs, explore=False)
                        if device == 'cuda':
                            agent_actions = [ac.cpu().data.numpy() for ac in torch_agent_actions]
                        else:
                            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                        actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                        next_obs, rewards, dones, infos = env.step(actions)
                        done = bool(sum([d for d in dones[0]])) or t >= config.episode_length
                        obs = next_obs
                        total_reward += np.sum(rewards)
                        t += 1
                print('Episode', ep_i, ': Evaluated total reward',
                      total_reward / n_test_episode / config.n_rollout_threads)
            else:
                print('Episode', ep_i, ': Episode reward', ep_rews)

        if fully_cooperative:
            logger.add_scalars('total_episode_rewards', {'total_rewards': sum(ep_rews)}, global_step=ep_i)
        else:
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalars('mean_episode_rewards', {f'agent_{a_i}': a_ep_rew}, global_step=ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

        # calculate based on info
        if config.env_id == 'simple_spread':
            episode_infos = np.average(np.array([info[0]['n'][0] for info in episode_infos]), axis=0)
            logger.add_scalars('episode_infos', {
                'collisions': episode_infos[1],
                'dist': episode_infos[2],
                'occupied': episode_infos[3],
            }, global_step=ep_i)

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    with open(str(log_dir) + 'args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_adversary', help="Name of environment")
    # simple_spread: cooperative navigation
    # simple_speaker_listener: cooperative communication
    parser.add_argument("--model_name",
                        default="maddpg",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=535, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=80000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024 * 25, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=1.0, type=float)
    parser.add_argument("--final_noise_scale", default=0.05, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_false')

    config = parser.parse_args()
    run(config)
