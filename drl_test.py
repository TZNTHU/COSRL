import numpy as np
from myenv.ECFST_Env import ECS_Env
from matplotlib import pyplot as plt
# from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_controlfortest import *
import time
exp = 60
n_net = 3
net = [256]*n_net
gamma = 0.99
actor_lr = 1E-4 # 1E-4
critic_lr = 1E-4 # 1E-4
alpha_lr = 1E-4
nstep_per_episode = 7 #1150 # 3024 #289 # 3025
batch_size = 256 # 2048 # 256 # 128
epoch = 500 #1000 # 100000
train_num = 5
test_num = 1 
buffer_size = 100000
#algo_name = "sac" # "sac"



if __name__ == '__main__':    
    # create policy
    for i in range(1):
        time_start = time.perf_counter()
        #policy = create_policy_sac(net, gamma, actor_lr, critic_lr, alpha_lr)
        policy = create_policy_ppo(net, gamma, actor_lr, critic_lr)
        policy = load_policy(policy, 1)
        # policy = create_policy_ddpg(net, gamma, actor_lr, critic_lr)
        policy.eval()
        test = DummyVectorEnv([lambda: ECS_Env() for _ in range(test_num)])
        test_collector = Collector(policy, test)
        test.reset()  
        test_collector.reset()
        result = test_collector.collect(n_episode=test_num)
        time_end=time.perf_counter()
        time_sum = time_end - time_start
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
        
        print(result["rews"].mean(), result["rews"].std())
        print(time_sum)
