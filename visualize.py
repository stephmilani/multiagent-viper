import os
import pickle
import sys
import time

import imageio

from utils.make_env import make_env

if __name__ == '__main__':
    run_id = int(sys.argv[1])
    env_id = 'simple_adversary'
    save_folder = f"./maviper/python/viper/results/{env_id}/max-depth4/run{run_id}/"
    with open(save_folder + 'final_policy.pk', 'rb') as f:
        students = pickle.load(f)
    gif_path = save_folder + 'gif/'
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    episode_length = 25
    env = make_env(env_id, discrete_action=True)
    obs = env.reset()
    done = [False, ]
    frames = []

    for i in range(episode_length):
        actions = [student.predict([obs[agent]]) for agent, student in enumerate(students)]
        actions = [a.reshape(-1) for a in actions]
        obs, reward, done, info = env.step(actions)
        frames.append(env.render('rgb_array')[0])
        if done[0]:
            break
    env.close()
    imageio.mimsave(gif_path + 'final_policy.gif', frames, duration=1 / 30)
