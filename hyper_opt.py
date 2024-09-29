"""
-*- coding: utf-8 -*-
@Author : Wei Yixin
@Time : 2024/7/17 10:29
@File : hyper_opt.py
optimization of hyperparameters
"""
import datetime
import os

import pandas as pd
import torch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch import nn
from torchvision import models

#from train_and_evaluate4 import train_and_evaluate

from drl_train import drl_train

# 设置超参数空间
# space = {
#     # 'batch_size_train': hp.choice('batch_size_train', [8, 12]),
#     'batch_size_train': hp.choice('batch_size_train', [12]),
#     # 'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2]),
#     # 'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3]),
#     'learning_rate': hp.choice('learning_rate', [1e-3]),
#     # 'step_size': hp.choice('step_size', [5, 10]),
#     'step_size': hp.choice('step_size', [10]),
#     # 'gamma': hp.choice('gamma', [0.1, 0.2, 0.25, 0.5]),
#     # 'gamma': hp.choice('gamma', [0.1, 0.2, 0.25]),
#     'gamma': hp.choice('gamma', [0.1]),
#     # 'patience': hp.choice('patience', [5, 10]),
#     'patience': hp.choice('patience', [5]),
#     'random_seed': hp.choice('random_seed', list(range(1, 100))),
#     # 'train_transforms': hp.choice('train_transforms', [True, False]),
#     'train_transforms': hp.choice('train_transforms', [False]),
#     # 'base_model': hp.choice('base_model', ['resnet50', 'resnet18', 'vgg16'])
#     # 'epochs': hp.choice('epochs', [3])
# }

space = {
    'alpha_lr': hp.choice('alpha_lr', [1e-3,1e-4,1e-5]),
    'gamma': hp.choice('gamma', [0.9,0.95,0.99]),
    'seed':hp.choice('seed',[0,20,42,55,88]),
    'w1': hp.choice('w1', [0.3]),
    'w2': hp.choice('w2', [0.3]),
    'w3': hp.choice('w3', [0.4]),
    'tankstorage_w': hp.choice('tankstorage_w', [1.0,1.5]),
    'product_w': hp.choice('product_w', [1.0,1.5]),
    'property_w': hp.choice('property_w', [1.0,1.5]),
}
# space = {
#     'batch_size_train': hp.choice('batch_size_train', [8, 12]),
#     # 'batch_size_train': hp.choice('batch_size_train', [12]),
#     # 'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2]),
#     # 'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3]),
#     'learning_rate': hp.choice('learning_rate', [1e-3]),
#     'step_size': hp.choice('step_size', [5, 10]),
#     # 'step_size': hp.choice('step_size', [10]),
#     # 'gamma': hp.choice('gamma', [0.1, 0.2, 0.25, 0.5]),
#     # 'gamma': hp.choice('gamma', [0.1, 0.2, 0.25]),
#     # 'gamma': hp.choice('gamma', [0.1, 0.25]),
#     'gamma': hp.choice('gamma', [0.1]),
#     # 'patience': hp.choice('patience', [5, 10]),
#     'patience': hp.choice('patience', [5]),
#     'random_seed': hp.choice('random_seed', list(range(1, 100))),
#     # 'train_transforms': hp.choice('train_transforms', [True, False]),
#     'train_transforms': hp.choice('train_transforms', [False]),
#     # 'base_model': hp.choice('base_model', ['resnet50', 'resnet18', 'vgg16'])
#     # 'epochs': hp.choice('epochs', [3])
# }

# 定义固定的hyperopt结果文件夹
parent_dir = os.path.abspath('..')
result_dir = os.path.join(parent_dir, 'results')
hyperopt_folder = os.path.join(result_dir, 'hyperopt_results')

# 创建hyperopt结果文件夹（如果不存在）
if not os.path.exists(hyperopt_folder):
    os.makedirs(hyperopt_folder)

# 获取当前时间并格式化
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 创建hyperopt结果文件夹中新的文件夹(基于当前时间）用于保存模型
experiment_dir = os.path.join(hyperopt_folder, f'results_{current_time}')
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)


# 定义优化函数
def objective(params):
    #base_model = 'resnet50'
    # model, train_losses, val_losses, results_df, mae_values, r2_values = drl_train(
    #     csv_file=os.path.abspath('..') + '\\data\\image_info_new_only.csv',
    #     root_dir=os.path.abspath('..') + '\\data\\images_new_only',
    #     #base_model=base_model,
    #     upper=5.0,
    #     mean=0.4065,
    #     std=0.091,
    #     ** params
    # )
    rew, lens, actor_loss, critic1_loss, critic2_loss, alpha_loss = drl_train(
        
    )

    # 保存每次超参数组合的结果到CSV文件
    params_results = {
        'Attempt': len(trials),
        #'base_model': base_model,
        #'batch_size_train': params['batch_size_train'],
        # 'learning_rate': params['learning_rate'],
        # 'step_size': params['step_size'],
        # 'gamma': params['gamma'],
        # 'patience': params['patience'],
        # 'random_seed': params['random_seed'],
        # 'train_transforms': params['train_transforms'],
        # 'min_val_loss': min(val_losses),
        # 'train_losses': train_losses,
        # 'val_losses': val_losses,
        # 'mae_values': mae_values,
        # 'r2_values': r2_values,
        'alpha_lr':params['alpha_lr'],
        'gamma':params['gamma'],
        'seed':params['seed'],
        'w1':params['w1'],
        'w2':params['w2'],
        'w3':params['w3'],
        'tankstorage_w':params['tankstorage_w'],
        'product_w':params['product_w'],
        'property_w':params['property_w'],
        'rew':rew,
        'lens':lens,
        'actor_loss':actor_loss,
        'critic1_loss':critic1_loss,
        'critic2_loss':critic2_loss,
        'alpha_loss':alpha_loss

    }

    epoch_results_df = pd.DataFrame([params_results])
    epoch_results_df.to_csv(os.path.join(experiment_dir, 'hyperopt_training_results.csv'), mode='a',
                            header=not os.path.exists(os.path.join(experiment_dir, 'hyperopt_training_results.csv')),
                            index=False,
                            )

    # 保存模型
    #model_file_path = os.path.join(experiment_dir, f'model_attempt_{len(trials)}.pth')
    #torch.save(model.state_dict(), model_file_path)

    # test_model = models.resnet50(weights=None)
    # test_model.fc = nn.Linear(test_model.fc.in_features, 1)
    # test_model.load_state_dict(torch.load(model_file_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    # test_model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # test_model.eval()
    #
    # def compare_models(model1, model2):
    #     for param1, param2 in zip(model1.parameters(), model2.parameters()):
    #         if not torch.equal(param1, param2):
    #             return False
    #     return True
    #
    # print("Model parameters are the same:", compare_models(test_model, model))

    #results_path = os.path.join(experiment_dir, f'model_attempt_{len(trials)}.csv')
    #results_df.to_csv(results_path, index=False)

    # 清理显存
    #del model
    #torch.cuda.empty_cache()
    print(len(critic1_loss),len(critic2_loss),len(rew))
    return {
        'loss': min((critic1_loss+critic2_loss)/2-rew),  # 将最小化的目标作为loss返回
        'params': params,
        'status': STATUS_OK,
        # 'model': model,
        'rew':rew,
        'lens':lens,
        'actor_loss':actor_loss,
        'critic1_loss':critic1_loss,
        'critic2_loss':critic2_loss,
        'alpha_loss':alpha_loss
    }


# 执行优化
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

# 将最优结果的超参数字典转换为字符串
best_trial = trials.best_trial['result']
# note = "Best hyperparameters: " + ", ".join([f"{key}: {value}" for key, value in best_trial['params'].items()])

# # 保存训练结果, 这里保存最优的结果，并更新到result文件夹的csv里面
# # 保证result文件夹只保存多次尝试的最好模型对应的结果
# # 用timestamp保证相同的文件名
# save_training_results(
#     best_trial['model'],
#     best_trial['train_losses'],
#     best_trial['val_losses'],
#     best_trial['results_df'],
#     best_trial['mae_values'],
#     best_trial['r2_values'],
#     note=note,
#     timestamp=current_time,
# )

print("Best hyperparameters found:")
print(best_trial['params'])
