import pandas as pd
import numpy as np

path_ = 'E://Machine learning/Database/H2/database_finish'
import sys
sys.path.append(path_)
from dataset import data
from mmoe import Model

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

import time
from train import TrainModel


if __name__ == '__main__':
    # ---------------建立数据集-----------------
    feature_ = 'Geometric+Energy'
    target_ = ['Adsorption_0.1bar', 'Adsorption_30bar', 'Adsorption_100bar',
               'Enthalpy_0.1bar', 'Enthalpy_30bar', 'Enthalpy_100bar']

    model_ = 'E://Machine learning/Database/H2/model_finish/'
    model_path = model_ + 'mmoe_' + feature_ + '_6task_COF_ALL.pth'

    x_train, y_train, x_type = \
        data('COF7-nom', feature_, target_, multitask_=True, feature_list=True).data_in()

    #x_test, y_test = \
        #data('COF7-nom', feature_, target_).data_in()

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                        test_size=0.2, random_state=10)
    # -------------相关参数-------------
    n_tasks = 6
    epochs = 20000
    #weight = [1, 1, 1]
    learningrate = 0.0006023097512118508
    MinTrainLoss = 1e10
    R2_pre = -1
    test_pre = [-1]
    train_pre = [-1]
    train_loss = []  # 储存损失

    device = torch.device("cuda")
    x_train = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    x_test = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)
    y_test = torch.from_numpy(y_test).to(device=device, dtype=torch.float32)

    Net = Model(num_feature=10, num_experts=9, units=10, expert_hidden=[7, 6],
                hidden_units=[9, 7], n_tasks=n_tasks).to(device=device)
    optimizer = torch.optim.Adam(Net.parameters(), lr=learningrate)
    loss_fun = nn.MSELoss()

    aoc_train, aoc_test, mse_test = TrainModel(Net, x_train, y_train,
                                        n_tasks, loss_fun, optimizer, epochs,
                                        x_test, y_test,
                                        save=True, save_name=model_path).train()
'''
    predict_all = Net(x_test)
    predict1 = predict_all[0].detach().cpu().numpy()
    predict1 = pd.DataFrame(predict1)
    predict1.to_csv('Pr1.csv', index=False)
    predict2 = predict_all[4].detach().cpu().numpy()
    predict2 = pd.DataFrame(predict2)
    predict2.to_csv('Pr2.csv', index=False)
    y_test = y_test.detach().cpu().numpy()
'''