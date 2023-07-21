

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


J = 10
NB_EPOCH = 1000
BATCH_SIZE = 128
DROPOUT_RATIO = 0.5

# F(t) = 1 - exp(-(t-g)^m/s)


def weibull_cdf(parameters: torch.Tensor):
    m = parameters[:, 0]
    s = torch.clamp(parameters[:, 1], min=0.001)
    output_list = []
    for num in range(J):
        Time = torch.Tensor([num])
        e_Time = torch.pow(Time, m)
        s_Time = torch.neg(torch.div(e_Time, s))
        # ref http://www.mogami.com/notes/weibull.html
        x = torch.subtract(torch.Tensor([1]), torch.exp(s_Time))
        output_list.append(x)
    return torch.stack(output_list, axis=1)

def weibull_cdf_debug(parameters: torch.Tensor):
    m = parameters[:, 0]
    s = torch.clamp(parameters[:, 1], min=0.001)
    output_list = []
    for num in range(J):
        Time = torch.Tensor([num])
        e_Time = torch.pow(Time, m)
        s_Time = torch.neg(torch.div(e_Time, s))
        # ref http://www.mogami.com/notes/weibull.html
        x = torch.subtract(torch.Tensor([1]), torch.exp(s_Time))
        output_list.append(x)
    return torch.stack(output_list, axis=1), m, s


class WeibullMLP(nn.Module):
    """
    WeibullMLP

    Weibull MLP from Nakagawa et al. 2020 (Brain Communications)

    Note: L1 and L2 regularization need to be implemented in Adam
    Optimizer separately 

    Parameters
    ----------
    nn :
        _description_


    loss = loss_fn(outputs, labels)

    # from https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch:

    l1_lambda = 0.001
    l1_norm = sum(torch.linalg.norm(p, 1) for p in model.parameters())

    loss = loss + l1_lambda * l1_norm

    So in_size will be batch_size x 150 and output is batch_size x 2
    """

    def __init__(self, in_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.do1 = nn.Dropout(DROPOUT_RATIO)
        self.do2 = nn.Dropout(DROPOUT_RATIO)
        self.do3 = nn.Dropout(DROPOUT_RATIO)
        self.s = nn.Linear(32, 1)
        self.sp = nn.Softplus()
        self.r = nn.Linear(32, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.do3(x)
        s = self.sp(self.s(x))
        r = F.relu(self.r(x))
        params = torch.cat([s, r], dim=1)
        return params
