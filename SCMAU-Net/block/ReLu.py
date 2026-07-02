import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Csigmoid(nn.Module):
    def __init__(self,in_ch):
        super(Csigmoid,self).__init__()
        self.in_ch = in_ch
        self.sig = nn.Sigmoid()

    def forward(self, x):
        img_real = x[:, :self.in_ch]
        img_imag = x[:, self.in_ch:]
        img_real_squ = img_real*img_real
        img_imag_squ = img_imag*img_imag
        C = img_real_squ + img_imag_squ
        C = C**0.5
        out = self.sig(C)
        return out


def magnitude(input):
    if input.ndimension() == 4:
        in_ch = input.shape[1]//2
        return (input[:, :in_ch] ** 2 + input[:, in_ch:] ** 2) ** (0.5)

class ModReLU(nn.Module):
    def __init__(self, in_channels, inplace=True):
        super(ModReLU, self).__init__()
        self.inplace = inplace
        self.in_channels = in_channels
        self.b = Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()
        self.relu = nn.ReLU(self.inplace)

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.0)

    def forward(self, x):
        mag = magnitude(x) + 1e-7
        if x.ndimension() == 4:
            brdcst_b = self.b.expand_as(mag)

        img = self.relu(mag+brdcst_b)/mag

        out = torch.cat([x[:,:self.in_channels]*img,x[:,self.in_channels:]*img],dim=1)
        del mag
        del img
        del brdcst_b
        return out
