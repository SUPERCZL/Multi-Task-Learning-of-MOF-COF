import time
import numpy as np

import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold

class TrainModel():
    def __init__(self, Net, x_train, y_train,
                        n_tasks, loss_fun, optimizer, epochs, #weight,
                        x_test=None, y_test=None,
                        save=None, save_name = None):
        self.Net = Net
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_tasks = n_tasks
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.epochs = epochs
        #self.weight = weight
        self.save = save
        self.save_name = save_name

    def kfold_train(self, SEED, N):
        kf = KFold(n_splits=N, random_state=SEED, shuffle=True)
        test_aoc_fold = []
        test_MSE_fold = []
        for i, (train_index, test_index) in enumerate(kf.split(self.x_train)):
            x_train_fold = self.x_train[train_index]
            y_train_fold = self.y_train[train_index]
            x_verify_fold = self.x_train[test_index]
            y_verify_fold = self.y_train[test_index]

            aoc_train, aoc_test, mse_test = \
            self.train(True, x_train_fold, y_train_fold, x_verify_fold, y_verify_fold)
            if i == 0:
                test_aoc_fold_buff = np.array(aoc_test)
            else:
                test_aoc_fold_buff += aoc_test
            test_aoc_fold.append(float(np.mean(aoc_test)))
            test_MSE_fold.append(float(np.mean(mse_test)))

        print(test_aoc_fold_buff/5)

        return np.mean(test_aoc_fold), np.mean(test_MSE_fold)

    def train(self, K=False, x_train=False, y_train=False, x_test=False, y_test=False):
        if not K:
            x_train = self.x_train
            y_train = self.y_train
            x_test = self.x_test
            y_test = self.y_test

        start = time.time()
        #start0 = time.time()
        for epoch in range(1, self.epochs + 1):
            self.Net.train()
            train_predit = self.Net(x_train)
            loss = 0
            #g=[1,1,1,1,10,100]
            for i in range(self.n_tasks):
                outputs = train_predit[i].view(-1)
                loss_buff = torch.sqrt(self.loss_fun(outputs, y_train[:, i]))
                loss += loss_buff#*g[i]

            self.optimizer.zero_grad()  # 在每一次迭代梯度反传更新网络参数时，需要把之前的梯度清0，不然上一次的梯度会累积到这一次。
            loss.backward()  # 反向传播
            self.optimizer.step()  # 优化更新

            # 记录误差，每1000个epoch保存一次loss
            if epoch % 1000 == 0:
                end = time.time()

                print("epoch:[%5d/%5d] time:%.2fs current_loss:%.5f"
                      % (epoch, self.epochs, (end - start), loss.item()))

                predict_all = self.Net(x_test)

                aoc_train = []
                aoc_test = []
                mse_test = []
                for i in range(self.n_tasks):
                    buff1 = train_predit[i].detach().cpu().numpy()
                    buff2 = y_train[:, i].detach().cpu().numpy()
                    aoc_train.append(round(r2_score(buff2, buff1), 6))
                    predict = predict_all[i].detach().cpu().numpy()
                    y_test_buff = y_test[:, i].detach().cpu().numpy()
                    aoc_test.append(round(r2_score(y_test_buff, predict), 6))
                    mse_test.append(mean_squared_error(y_test_buff, predict))
                    print('R2_train%s:'%(i+1), aoc_train[i], 'R2_test%s:'%(i+1), aoc_test[i])
        #end0 = time.time()

        if self.save:
            torch.save(self.Net.state_dict(), self.save_name)

        return aoc_train, aoc_test, mse_test