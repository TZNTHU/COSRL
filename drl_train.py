import numpy as np
from myenv.ECFS_Env import ECS_Env
from matplotlib import pyplot as plt
# from conventional_control import ConventionalCTRL
from tianshou.data import Batch
from copy import deepcopy
from drl_control import *

# exp = 66
# n_net = 3
# net = [256]*n_net
# gamma = 0.99
# actor_lr = 1E-4 # 1E-4  
# critic_lr = 1E-4 # 1E-4
# alpha_lr = 1E-4  
# nstep_per_episode = 7 #1150 # 3024 #289 # 3025
# batch_size = 256 # 2048 # 256 # 128
# epoch = 100 #1000 # 100000
# train_num = 5
# test_num = 10
# buffer_size = 100000
# algo_name = "sac" # "sac" 

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

# 在训练脚本的开头设置随机种子

# if __name__ == '__main__':    
#     # create policy
#     # policy = create_policy_ddpg(net, gamma, actor_lr, critic_lr, alpha_lr,auto_alpha=True)
#     #for i in range(3):
#     policy = create_policy_sac(net, gamma, actor_lr, critic_lr, alpha_lr)
#     # policy = load_policy(policy, 19)
#     # policy = create_policy_ddpg(net, gamma, actor_lr, critic_lr)

#     rew_avg, rew_std = train_policy(policy, exp , algo_name, epoch, nstep_per_episode, batch_size,train_num, test_num, buffer_size)

def drl_train(exp=68,gamma = 0.99,actor_lr = 1e-5,critic_lr = 1e-5,alpha_lr = 1e-4,batch_size = 256,epoch = 2000,train_num = 5,test_num = 10,seed = 42,w1 = 0.3,w2 = 0.3,w3 = 0.0,tankstorage_w = 1.5,product_w = 1.0,property_w = 1.5):
    n_net = 3
    net = [256]*n_net
    nstep_per_episode = 7
    buffer_size = 100000
    algo_name = "sac"
    set_seed(seed)
    w = [1.0,tankstorage_w,1.0,product_w,1.0,property_w,1.0,1.0,1.0]
    policy = create_policy_sac(net, gamma, actor_lr, critic_lr, alpha_lr,seed=seed,w1=w1,w2=w2,w3=w3,weight=w,auto_alpha=False)
    rew_avg, rew_std, reward, lens, actor_loss, critic1_loss, critic2_loss,rew = train_policy(policy, exp , algo_name, epoch, nstep_per_episode, batch_size,train_num, test_num, buffer_size,seed= seed)
    
    return rew, lens, actor_loss, critic1_loss, critic2_loss

# if __name__ == '__main__':
#     for i in range(5):
#         #seed_list = [0,34,51]
#         a = np.random.randint(0,100)
#         rew, lens, actor_loss, critic1_loss, critic2_loss= drl_train(exp=97+i,seed=a,epoch=1000,w3=0.0)
if __name__ =='__main__':
    rew, lens, actor_loss, critic1_loss, critic2_loss= drl_train(exp=102,seed=0,epoch=1000,w3=0.0)
    #print(len(critic1_loss),len(critic2_loss),len(rew))
    #print(critic1_loss[-1],critic2_loss[-1],rew[-1])
    #print(critic1_loss,critic2_loss,rew)
# if __name__ == '__main__':
#     for i in range(3):
#         seed_list = [0,35,50]
#         _,_,_,_,_,_ = drl_train(exp=66+i,seed=seed_list[i])


#查看训练曲线：在终端输入“tensorboard --logdir="路径"”回车即可。“路径”指向本python文件所处文件夹的log文件，例：E:\project\reinforcement\tianshou-0.4.8\log

#29 增加权重值分别为10和2，fail
#30 逐步增加惩罚项至10~50和2 500：fail
#33 尝试使用硬约束
#36 w1 w2 w3 0.25 0.3 0.5 1.2change_penalty 0brine_penalty
#75
#90
#94 切换次数太多
#96 9次切换

