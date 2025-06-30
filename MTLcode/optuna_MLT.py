import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

path_ = 'E://Machine learning/Database/H2/database_finish'
import sys
sys.path.append(path_)

from train import TrainModel
from dataset import data
from share_model import Model
import optuna


def objective(trial):
    n_tasks = 3
    num_feature = 10
    #MinTrainLoss = 1e10

    lr = trial.suggest_float('lr', 0.00001, 0.01, log=True)  # 整数型，(参数名称，下界，上界，步长)
    num_experts = trial.suggest_int('num_experts', 4, 12)

    expert_hidden1 = trial.suggest_int('expert_hidden1', 2, 12)
    expert_hidden2 = trial.suggest_int('expert_hidden2', 2, 12)
    expert_hidden3 = trial.suggest_int('expert_hidden3', 2, 12)
    expert_hidden=[expert_hidden1, expert_hidden2, expert_hidden3]

    hidden_units1 = trial.suggest_int('hidden_units1', 2, 12)
    hidden_units2 = trial.suggest_int('hidden_units2', 2, 12)
    hidden_units3 = trial.suggest_int('hidden_units3', 2, 12)
    hidden_units = [hidden_units1, hidden_units2]


    units = trial.suggest_int('units', 4, 12)
    epochs = trial.suggest_categorical('epochs', [10000, 20000, 30000])

    Net = Model(num_feature=num_feature, num_experts=num_experts,
                units=units, expert_hidden=expert_hidden,
                hidden_units=hidden_units, n_tasks=n_tasks).to(device=device)
    optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
    loss_fun = nn.MSELoss()

    test_aoc, mse_aoc = TrainModel(Net, x_train, y_train,
                                   n_tasks=n_tasks, loss_fun=loss_fun,
                                   optimizer=optimizer, epochs=epochs).kfold_train(10, 5)

    return test_aoc

if __name__ == '__main__':
    # ---------------建立数据集-----------------
    feature_ = 'Geometric+Energy'
    target_ = ['Enthalpy_0.1bar', 'Enthalpy_30bar', 'Enthalpy_100bar']

    x_train, y_train, x_type = \
        data('MOF-nom', feature_, target_, multitask_=True, feature_list=True).data_in()
    '''
    x_train2, y_train2 = \
        data('COF7-nom', feature_, target_).data_in()
    '''
    device = torch.device("cuda")
    x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    #x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)

    # 创建Optuna study
    study = optuna.create_study(direction='maximize')

    # 运行Optuna搜索
    study.optimize(objective, n_trials=50)

    # 打印最佳超参数和得分
    print(feature_+',share,MOF')
    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)