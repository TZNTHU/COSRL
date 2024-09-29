import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import DQNPolicy, A2CPolicy, PPOPolicy, DiscreteSACPolicy
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer, onpolicy_trainer, offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.discrete import Actor, Critic
from examples.atari.atari_network import DQN
from torch import nn

from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from myenv.ECFS_Env import ECS_Env

# create DQN policy
def create_policy_dqn(hidden_sizes=[64,64], gamma=0.99, lr=1E-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    policy = DQNPolicy(
        net,
        optim,
        action_space=action_shape,
        observation_space=state_shape,
        discount_factor=gamma,
        estimation_step=1,
        target_update_freq=0,
        reward_normalization=False,
        is_double=True,
    )
    return policy

# create A2C policy
def create_policy_a2c(hidden_sizes=[64,64], gamma=0.99, lr=1E-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    net = Net(
        state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    ).to(device)
    actor = Actor(net, action_shape, device=device, softmax_output=False)
    critic = Critic(net, device=device)
    actor_critic = ActorCritic(actor,critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=1e-5)

    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = A2CPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=env.action_space,
        observation_space=env.observation_space,
    ).to(device) 
    return policy


# create PPO policy
def create_policy_ppo(hidden_sizes=[64,64], gamma=0.99, gae_lambda=0.95,max_grad_norm=0.5,vf_coef=0.25,ent_coef=0.01,eps_clip=0.1,lr=1E-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    # define model
    # net = DQN(
    #     *state_shape,
    #     action_shape,
    #     device=device,
    #     features_only=True,
    #     output_dim=hidden_sizes
    # ).to(device)
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    ).to(device)
    
    actor = Actor(net, action_shape, device=device, softmax_output=False)
    critic = Critic(net, device=device)
    actor_critic = ActorCritic(actor,critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=1e-5)

    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        reward_normalization=False,
        action_scaling=False,
        lr_scheduler=None,
        action_space=env.action_space,
        eps_clip=eps_clip,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=1,
        recompute_advantage=0,
    ).to(device)

    return policy

# create SAC policy
def create_policy_sac(hidden_sizes=[64,64],tau=0.005 , gamma=0.99, alpha=0.05, n_step=3, actor_lr=1E-5,critic_lr=1E-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = ECS_Env()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_shape=np.prod(action_shape)
    # define model
    # net = DQN(
    #     *state_shape,
    #     96,
    #     96,
    #     action_shape,
    #     device=device,
    #     features_only=True,
    #     output_dim=512,
    #     )
    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    ).to(device)
    actor = Actor(net, action_shape, device=device, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic1 = Critic(net, last_size= action_shape, device=device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net, last_size=action_shape, device=device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    policy = DiscreteSACPolicy(
    actor=actor,
    actor_optim=actor_optim,
    critic1=critic1,
    critic1_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    action_space=env.action_space,
    tau=tau,
    gamma=gamma,
    alpha=alpha,
    estimation_step=n_step,
    ).to(device)

    return policy


# train policy
def train_off_policy(policy, experiment, algo_name, epoch, nstep_per_episode=1000, batch_size=1024*2, train_num=1, test_num=5, buffer_size=1000000):
    
    train_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(train_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(test_num)]
    )
    buffer = VectorReplayBuffer(buffer_size, train_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    train_collector.collect(n_episode=20, random=True)
    
    log_name = os.path.join("ECFS-v11", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
    
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=nstep_per_episode,
        step_per_collect=10,
        episode_per_test=test_num,
        batch_size=batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=1,
        test_in_train=False,
    ).run()

    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(0)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std()

def train_on_policy(policy, experiment, algo_name, epoch, nstep_per_episode=1000, batch_size=1024*2, train_num=1, test_num=5, buffer_size=1000000):
    
    train_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(train_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: ECS_Env() for _ in range(test_num)]
    )
    buffer = VectorReplayBuffer(buffer_size, train_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_episode=20, random=True)
    
    log_name = os.path.join("ECFS-v11", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
    

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epoch,
        nstep_per_episode,
        1,
        test_num,
        batch_size,
        1000,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    ).run()
    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(0)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std()

# load policy
def load_policy(policy, algo_name, experiment):
    
    # path
    # algo_name = "dqn"
    log_name = os.path.join("ECFS-v11", algo_name, str(experiment))
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
    log_name = os.path.join("ECFS-v11", algo_name, str(experiment))
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