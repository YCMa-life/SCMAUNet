import torch
import torch.nn as nn

class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        self.in_channel=in_channel

        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):

        real = self.conv_re(x[:,:self.in_channel]) - self.conv_im(x[:,self.in_channel:])
        imaginary = self.conv_re(x[:,self.in_channel:]) + self.conv_im(x[:,:self.in_channel])
        output = torch.cat([real,imaginary],dim=1)
        del(real,imaginary)
        return output

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(ComplexConvTranspose2d,self).__init__()
        self.in_channel = in_channel

        self.rConvTranspose2d = nn.ConvTranspose2d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups,
                                bias=bias)

        self.iConvTranspose2d = nn.ConvTranspose2d(in_channel, out_channel, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups,
                                bias=bias)

    def forward(self, x):
        real = self.rConvTranspose2d(x[:,:self.in_channel]) - self.iConvTranspose2d(x[:,self.in_channel:])
        imaginary = self.rConvTranspose2d(x[:,self.in_channel:]) + self.iConvTranspose2d(x[:,:self.in_channel])

        out = torch.cat([real,imaginary],dim=1)
        del(real,imaginary)
        return out