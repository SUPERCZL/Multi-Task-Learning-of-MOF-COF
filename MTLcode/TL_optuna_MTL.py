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
from TL_share_model import Model
import optuna

def load_model(model_path,
               num_feature, num_experts, units, expert_hidden,
               hidden_units, n_tasks):
    device = torch.device("cuda")
    Model_load = Model(num_feature=num_feature,
                       num_experts=num_experts,
                       units=units,
                       expert_hidden=expert_hidden,
                       hidden_units=hidden_units,
                       n_tasks=n_tasks).to(device=device)
    Model_load.load_state_dict(torch.load(model_path))
    return Model_load

def split_data(x, y, num_train, seed, index_number=None):
    '''
    num_train:切分数据大小
    index_number:预先抽样的list
    '''
    index_array = list(range(len(x)))
    if index_number:
        index_number = index_number[:num_train]
    else:
        random.seed(seed)
        index_number = random.sample(index_array, num_train)
    x_train = x[index_number]
    y_train = y[index_number]
    for i in index_number:
        index_array.remove(i)
    x_test = x[index_array]
    y_test = y[index_array]
    return x_train, y_train, x_test, y_test

def objective(trial):
    n_tasks = 3
    num_feature = 10
    #MinTrainLoss = 1e10

    lr = trial.suggest_float('lr', 0.00001, 0.6, log=True)  # 整数型，(参数名称，下界，上界，步长)

    expert_hidden = [10, 10, 10]
    hidden_units = [12, 7]

    epochs = trial.suggest_categorical('epochs', [5000, 10000, 15000, 20000])

    #---------------load model---------------
    model_ = 'E://Machine learning/Database/H2/model_finish/'
    model_path = model_ + 'share_' + feature_ + '_MOF_base_ALL.pth'
    Net = load_model(model_path,
                     num_feature, 10, 11, expert_hidden,
                     hidden_units, n_tasks
                     )

    optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
    loss_fun = nn.MSELoss()

    test_aoc, mse_aoc = TrainModel(Net, x_train, y_train,
                                   n_tasks=n_tasks, loss_fun=loss_fun,
                                   optimizer=optimizer, epochs=epochs).kfold_train(10, 5)

    return test_aoc

if __name__ == '__main__':
    # ---------------建立数据集-----------------
    feature_ = 'Geometric+Energy'
    target_ = ['Work_12', 'Work_23']

    x_train, y_train, x_type = \
        data('MOF-nom', feature_, target_, multitask_=True, feature_list=True).data_in()

    x_test, y_test = \
        data('COF7-nom', feature_, target_, multitask_=True).data_in()

    # ------------build fine-tuning data------------
    SEED = 15
    x_train, y_train, x_test, y_test = split_data(x_test, y_test, 400, SEED)

    device = torch.device("cuda")
    x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    #x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)

    # 创建Optuna study
    study = optuna.create_study(direction='maximize')

    # 运行Optuna搜索
    study.optimize(objective, n_trials=50)

    # 打印最佳超参数和得分
    print(feature_+'_TL,share')
    print('Best hyperparameters: ', study.best_params)
    print('Best score: ', study.best_value)