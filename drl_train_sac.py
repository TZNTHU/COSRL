import numpy as np
from myenv.ECFS_Env import ECS_Env
from matplotlib import pyplot as plt
# from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_schedule import *

exp = 200
n_net = 3
net = [256]*n_net
tau = 0.005
gamma = 0.99
alpha = 0.05
n_step = 3
actor_lr = 1E-4 # 1E-4
critic_lr = 1E-4

nstep_per_episode = 1000 # 3024# 289 # 3025
batch_size = 256 # 2048 # 256 # 128
epoch = 10000 # 100000
train_num = 2 # 2
test_num = 5 # 5
buffer_size = 100000
algo_name = "sac" # "sac"

if __name__ == '__main__':    
    # create policy
    policy = create_policy_sac(net, tau, gamma, alpha, n_step, actor_lr,critic_lr)

    rew_avg, rew_std = train_off_policy(policy, exp, algo_name, epoch, nstep_per_episode, batch_size,train_num, test_num, buffer_size)