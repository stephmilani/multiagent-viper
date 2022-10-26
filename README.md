# MAVIPER: Learning Decision Tree Policies for Interpretable Multi-Agent Reinforcement Learning

This is the code base for the paper [MAVIPER: Learning Decision Tree Policies for Interpretable Multi-Agent Reinforcement Learning
](https://arxiv.org/abs/2205.12449).

## Getting started
- Create a new virtual environment using `environment.yml`. We also provide a `setup.sh` which we have used to run the code locally.
- Add `maviper` and `maviper/python/viper` to `PYTHONPATH`

## Running the code
- To generate a compatible neural network model, please run the included MADDPG implementation, adapted from [MADDPG-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch).
- To run MAVIPER and IVIPER, use `python -u maviper/python/viper/train_students.py`.

The details of the arguments can be found within `train_students.py`.
  - MAVIPER is used by default. Add `--not_joint` for IVIPER.
  - An example would be `xvfb-run --auto-servernum --server-num=1 python -u maviper/python/viper/train_students.py --test_gen --max_depth 4 --n_batch_rollouts 50 --max_iters 100 --joint_training --test_name joint_dt --selection_metric reward --estimate_by_team --average_others --is_train --random_seed 555 --scenario_name {scenario_name} --model_path {PATH}`.
- The environments are `simple_adversary`, `simple_tag` and `simple_spread` of the [MPE](https://github.com/openai/multiagent-particle-envs) environment.

## Acknowledgements
Our implementation is based on [VIPER](https://github.com/obastani/viper) and [MADDPG-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch).