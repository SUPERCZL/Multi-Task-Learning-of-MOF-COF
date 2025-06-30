import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.3)
        # self.log_soft = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.log_soft(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        #self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[1], output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        #self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
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
        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(self.input_size, self.num_experts), requires_grad=True) for i in range(self.tasks)])
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        # get the experts output
        expers_o = [e(x) for e in self.experts]
        expers_o_tensor = torch.stack(expers_o)

        # get the gates output
        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        # multiply the output of the experts with the corresponding gates output
        towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor for g in gates_o]
        towers_input = [torch.sum(ti, dim=0) for ti in towers_input]

        # get the final output from the towers
        final_output = [t(ti) for t, ti in zip(self.towers, towers_input)]

        # get the output of the towers, and stack them
        #final_output = torch.cat(final_output, dim=1)
        #print(final_output.shape)
        #print(background.shape)

        return final_output