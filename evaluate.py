import argparse
import os
import sys
from multiprocessing import Pool

import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG


def run_model(model_path, gif_path, config):
    result, all_infos = [], []
    maddpg = MADDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=maddpg.discrete_action)
    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval

    total_rewards = []
    total_infos = []
    for ep_i in range(config.n_episodes):
        # print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        if config.save_gifs:
            env.render('human')
        episode_rewards = []
        episode_infos = []
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            episode_infos.append(infos['n'])
            episode_rewards.append(rewards)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render('human')
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

        total_rewards.append(np.sum(episode_rewards, axis=0))

        if config.env_id == 'simple_tag':
            total_infos.append(np.sum([e[0][0] + e[1][0] for e in episode_infos]))
        else:
            total_infos.append(np.sum(np.array(episode_infos), axis=0))

    if config.env_id == 'simple_spread':
        result.append(np.sum(np.mean(total_rewards, axis=0)))
    else:
        total_rewards = np.mean(total_rewards, axis=0)
        total_infos = np.mean(total_infos, axis=0)
        result.append(np.sum(total_rewards, axis=0))
        all_infos.append(np.sum(total_infos, axis=0))
    env.close()
    return result, all_infos


def run(config):
    model_path = (Path('./models') / config.env_id / ('run%i' % config.run_num))
    # sys.stdout = open(str(model_path / 'evaluation_result.txt'), 'w')
    model_paths = []
    if config.load_all:
        model_paths = [model_path / 'incremental' / i.name for i in os.scandir(model_path / 'incremental')]
        # if os.path.exists(model_path / 'model.pt'):
        #     model_paths.append(model_path / 'model.pt')
    elif config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'
    if len(model_paths) == 0:
        model_paths.append(model_path)

    gif_path = None
    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    result = []
    all_infos = []
    model_paths = sorted(model_paths, key=lambda x: int(x.name.split('_')[1].split('.')[0][2:]))[-200:]
    with Pool(32) as p:
        for r, i in p.starmap(run_model, [(m, gif_path, config) for m in model_paths]):
            result += r
            all_infos += i

    if config.env_id == 'simple_tag':
        # Choose the predator with the most catches
        choose = np.argmax(all_infos)
    else:
        # For fully cooperative envs, choose by the sum of reward
        choose = np.argmax(result)
    print(f'The best policy is {model_paths[np.argmax(result)]}, with reward {result[choose]}')
    if config.env_id == 'simple_tag':
        print('Average # of collisions: %f' % all_infos[choose])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment")
    parser.add_argument("--model_name", default='maddpg',
                        help="Name of model")
    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--load_all", action="store_true")
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
