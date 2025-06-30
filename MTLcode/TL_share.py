import torch
import torch.nn as nn

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
class Share(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Share, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        #freeze(self)
        self.fc4 = nn.Linear(hidden_size[2], output_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.3)
        #self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        #out = self.log_soft(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.4)
        #self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.softmax(out)
        # out = torch.sigmoid(out)
        return out

class Model(torch.nn.Module):
    def __init__(self, **config):
        super(Model, self).__init__()
        # params
        self.input_size = config['num_feature']
        self.num_experts = config['num_experts']
        self.experts_out = config['units']
        self.experts_hidden = config['expert_hidden']
        self.towers_hidden = config['hidden_units']
        self.tasks = config['n_tasks']
        # row by row
        self.softmax = nn.Softmax(dim=1)
        # model
        self.experts = nn.Sequential(Share(self.input_size, self.experts_out, self.experts_hidden))
        freeze(self.experts)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        out_1 = self.experts(x)
        final_output = [ti(out_1) for ti in self.towers]
        return final_output