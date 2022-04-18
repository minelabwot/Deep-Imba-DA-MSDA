import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x = x.view(-1,10*32)
        x = x.view(-1,64)
        # print('log68',x.size())
        # print('log16',x.size())
        x = F.relu(self.dis1(x))
        # print('log10',x.size())

        # x = self.dis2(self.bn(x))
        x = self.dis2(x)
        # print('log99',x.size())
        x = torch.sigmoid(x)
        # print('log100',x.size())
        # print('log101',x)
        return x
