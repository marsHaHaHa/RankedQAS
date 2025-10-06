import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
            num_layers:
            Input_dim:
            hidden_dim:
            Output_dim:
        '''
        super(MLP, self).__init__()
        self.linear_or_not = False  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim))) #中间的隐层需要归一化

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class GIN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = 64
        self.output_dim = 1
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(3, self.input_dim, self.hidden_dim, self.hidden_dim))
            else:
                self.mlps.append(MLP(3, self.hidden_dim, self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.output_dim)

    def forward(self, data):
        adj = data.adj.view(data.num_graphs, -1, data.adj.shape[-1])
        adj = ((adj + adj.transpose(-1, -2)) > 0).to(adj.dtype)
        ops = data.x.view(data.num_graphs, -1, data.num_node_features)
        batch_size, node_num, opt_num = ops.shape
        x = ops #37*13
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)  # agg: 16*37=592，13
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))  # 16*37*128
        out = self.fc1(x)
        Z = torch.sum(out, dim=-2)
        return Z



