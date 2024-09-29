import argparse
import datetime
import os
import pprint

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import DDPGPolicy, SACPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.exploration import GaussianNoise
from torch.distributions import Independent, Normal

from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from myenv.ECFST_Env import ECS_Env


# create policy PPO 
def create_policy_ppo(hidden_sizes = [64,64],gamma = 0.99,actor_lr = 1e-4,critic_lr = 1e-4,device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env(random_seed_ref=random.randint(0,10000))
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    net_a = Net(state_shape=state_shape,hidden_sizes=hidden_sizes,activation=torch.nn.Tanh,device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    #actor_optim = torch.optim.Adam(actor.parameters(),lr = actor_lr)
    net_c = Net(
        state_shape=state_shape,
        action_shape= action_shape,
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        #concat=True,
        device=device,
    )
    critic = Critic(net_c,device= device).to(device)
    #torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            #torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    #critic_optim = torch.optim.Adam(critic.parameters(),lr = critic_lr)
    optim = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr = actor_lr)

    def dist(*logits):
        return Independent(Normal(*logits), 1)
    
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist_fn=dist,
        discount_factor=gamma,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.1,
        ent_coef=0.01,
        #reward_normalization=True,
        action_scaling=True,
        action_bound_method='clip',
        gae_lambda=0.95,
        action_space=env.action_space      
    )

    return policy

# create policy
def create_policy_sac(hidden_sizes=[64,64], gamma=0.99, actor_lr=1E-4, 
                  critic_lr=1E-4, alpha_lr=3E-4, auto_alpha=True, 
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)
    
    if auto_alpha:
        alpha_lr = alpha_lr
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = 0.5


    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        # tau=0.01,
        gamma=gamma,
        alpha=alpha,
        # reward_normalization=True,
        estimation_step=1,
        action_space=env.action_space,
    )
    
    return policy

def create_policy_ddpg(hidden_sizes=[64,64], gamma=0.99, actor_lr=1E-4, 
                  critic_lr=1E-4, tau=0.005, exploration_noise=0.1,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]

    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic = Critic(net_c, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    policy = DDPGPolicy(
        actor,
        actor_optim,
        critic,
        critic_optim,
        tau=tau,
        gamma=gamma,
        exploration_noise=GaussianNoise(sigma=exploration_noise), # for training
        # exploration_noise = None, # for testing
        estimation_step=1,
        action_space=env.action_space,
    )
    return policy

# train policy
def train_policy(policy, experiment, algo_name, epoch, nstep_per_episode=1150, batch_size=1024*2, train_num=1, test_num=20, buffer_size=1000000):
    
    train_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(train_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(test_num)]
    )
    
    # buffer = ReplayBuffer(buffer_size)
    buffer = VectorReplayBuffer(buffer_size, train_num)
    
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_episode=20, random=True)
    
    log_name = os.path.join("PenSim-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        
    result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            epoch,
            nstep_per_episode,
            1,
            test_num,
            batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=1,
            test_in_train=False,
        )
    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(0)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std()

# load policy
def load_policy(policy, experiment):
    
    # path
    #algo_name = "sac"
    algo_name = 'ppo'
    log_name = os.path.join("PenSim-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    # load from existing policy
    print(f"Loading agent under {log_path}")
    policy_path = os.path.join(log_path, "policy.pth")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path))
        print("Successfully loaded policy.")
    else:
        print("Fail to load policy.")
    
    return policy

# load policy
def load_policy_cpu(policy, algo_name, experiment):
    
    # path
    log_name = os.path.join("PenSim-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    # load from existing policy
    print(f"Loading agent under {log_path}")
    policy_path = os.path.join(log_path, "policy.pth")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
        print("Successfully loaded policy.")
    else:
        print("Fail to load policy.")
    
    return policy