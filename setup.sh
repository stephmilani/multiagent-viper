pwd=$PWD
sudo apt update && sudo apt full-upgrade
sudo apt install libopenmpi-dev
conda env create -f environment.yml
conda activate maviper
  export PYTHONPATH=$pwd/MAVIPER:$pwd/maviper/python/viper:$PYTHONPATH
cd multiagent-particle-envs/
pip install -e .
pip uninstall baselines
pip install baselines ipdb tensorboardX sortedcontainers ete3


conda activate maviper; pwd=$PWD; export PYTHONPATH=$pwd/MAVIPER:$pwd/maviper/python/viper:$PYTHONPATH