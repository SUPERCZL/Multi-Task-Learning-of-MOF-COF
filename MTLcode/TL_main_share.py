path_ = 'E://Machine learning/Database/H2/database_finish'
import sys
sys.path.append(path_)

import pandas as pd
import numpy as np
import random

from dataset import data
from TL_share import Model

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train import TrainModel

def load_model(model_path,
               num_feature, num_experts, units, expert_hidden,
               hidden_units, n_tasks):
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

if __name__ == '__main__':
    # ---------------建立数据集-----------------
    feature_ = 'Geometric+Energy'
    target_ = ['Adsorption_0.1bar', 'Adsorption_30bar', 'Adsorption_100bar']

    x_train, y_train, x_type = \
        data('MOF-nom', feature_, target_, multitask_=True, feature_list=True).data_in()

    x_test, y_test = \
        data('COF7-nom', feature_, target_, multitask_=True).data_in()

    # ------------build fine-tuning data------------
    SEED = 10
    x_train, y_train, x_test, y_test = split_data(x_test, y_test, 300, SEED)

    # -------------Parameters-------------
    n_tasks = 3
    learningrate = 0.0007531583936087021
    epochs = 15000

    num_experts = 5
    units = 10
    hidden_units = [10, 11]
    expert_hidden = [8, 7, 11]

    #---------------load model---------------
    model_ = 'E://Machine learning/Database/H2/model_finish/'
    model_path = model_ + 'share_' + feature_ + '_MOF_base_ALL.pth'
    device = torch.device("cuda")
    Net = load_model(model_path,
                     10, num_experts, units, expert_hidden,
                     hidden_units, n_tasks
                     )

    for k, v in Net.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)
    y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(Net.parameters(), lr=learningrate)
    loss_fun = nn.MSELoss()

    model_out = model_ + 'TL_share_' + feature_ + '.pth'
    aoc_train, aoc_test, mse_test = TrainModel(Net,
                                        x_train, y_train,
                                        n_tasks, loss_fun, optimizer, epochs,
                                        x_test=x_test, y_test=y_test,
                                        save=True, save_name=model_out).train()

    predict_all = Net(x_test)
    predict1 = predict_all[0].detach().cpu().numpy()
    predict2 = predict_all[1].detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
