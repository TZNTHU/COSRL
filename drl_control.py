import argparse
import datetime
import os
import pprint
import torch.nn.functional as F
import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import DDPGPolicy, SACPolicy
from tianshou.trainer import offpolicy_trainer,onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.exploration import GaussianNoise

from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from myenv.ECFS_Env import ECS_Env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保所有的操作都是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# create policy
def create_policy_sac(hidden_sizes=[64,64], gamma=0.99, actor_lr=1E-4, 
                  critic_lr=1E-4, alpha_lr=3E-4, seed = 42,w1 = 0.3,w2 = 0.3,w3 = 0.0,
                  weight = None,auto_alpha=True, 
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    set_seed(seed=seed)
    env = ECS_Env(random_seed_ref=seed,weight1=w1,weight2=w2,weight3=w3,weight=weight)
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
        #tau=0.01,
        gamma=gamma,
        alpha=alpha,
        #reward_normalization=True,
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
def train_policy(policy, experiment, algo_name, epoch, nstep_per_episode=1150, batch_size=1024*2, train_num=1, test_num=20, buffer_size=1000000,seed = 42):
    
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
    
    actor_loss_values = []
    critic1_loss_values = []
    critic2_loss_values = []
    #alpha_loss_values = []
    rewls = []
    

    def save_checkpoint_fn(epoch,env_step,gradient_step):
        
        batch, indices = train_collector.buffer.sample(batch_size)
        # batch.returns = compute_returns(batch, 0.99)
        batch = policy.process_fn(batch,buffer,indices)
        loss = policy.learn(batch)
        # # #loss = policy._last_loss

        actor_loss_values.append(loss['loss/actor'])
        critic1_loss_values.append(loss['loss/critic1'])
        critic2_loss_values.append(loss['loss/critic2'])
        #alpha_loss_values.append(loss['loss/alpha'])
    
    def save_reward_fn(rew):
        rewls.append(rew.mean())
        return rew
        
    
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
            #callback_fn = callback_fn
            save_checkpoint_fn = save_checkpoint_fn,
            reward_metric = save_reward_fn
        )
    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(42)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std(),result["rews"],result["lens"],actor_loss_values,critic1_loss_values,critic2_loss_values,rewls

def train_policy_on(policy, experiment, algo_name, epoch, nstep_per_episode=1150, batch_size=1024*2, train_num=1, test_num=20, buffer_size=1000000):
    
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
    train_collector.collect(n_episode=20, random=False)
    
    log_name = os.path.join("PenSim-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        
    #n_step = 7
    result = onpolicy_trainer(
            policy = policy,
            train_collector = train_collector,
            test_collector =test_collector,
            max_epoch = epoch,
            step_per_epoch = nstep_per_episode,
            repeat_per_collect = 1,
            episode_per_test = test_num,
            step_per_collect = 5,
            batch_size = batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=1,
            test_in_train=False,

        )
    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(42)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std()
# load policy
def load_policy(policy, experiment):
    
    # path
    algo_name = "sac"
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