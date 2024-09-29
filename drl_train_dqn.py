import numpy as np
from myenv.ECFS_Env import ECS_Env
from matplotlib import pyplot as plt
# from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_schedule import *

exp = 1
n_net = 3
net = [256]*n_net
gamma = 0.99
lr = 1E-5 # 1E-4
nstep_per_episode = 1000 # 3024# 289 # 3025
batch_size = 256 # 2048 # 256 # 128
epoch = 15000 # 100000
train_num = 2 # 2
test_num = 5 # 5
buffer_size = 100000
algo_name = "dqn" # "sac"

if __name__ == '__main__':    
    # create policy
    policy = create_policy_dqn(net, gamma, lr)
    # policy = load_policy(policy, "dqn",1)

    rew_avg, rew_std = train_off_policy(policy, exp, algo_name, epoch, nstep_per_episode, batch_size,train_num, test_num, buffer_size)