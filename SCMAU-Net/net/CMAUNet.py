import torch
import torch.nn as nn
from block.Cbn import ComplexBatchNorm2d
from block.C_conv import ComplexConv, ComplexConvTranspose2d
from block.ReLu import ModReLU, Csigmoid


# CBAM: Convolutional Block Attention Module
# Reference: Woo, S., Park, J., Lee, J.Y., Kweon, I.S. "CBAM: Convolutional
# Block Attention Module", ECCV 2018.
# Code adapted from: https://github.com/Jongchan/attention-module

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        in_ch = in_planes*2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        ca = self.channel_attention(x)
        out = x * ca
        sa = self.spatial_attention(out)
        out = out * sa
        return out

# Reference: Zhou Y, Kong Q, Zhu Y, et al. "MCFA-UNet: Multiscale cascaded feature
# attention U-Net for liver segmentation", IRBM 2023.

class general_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(general_conv, self).__init__()
        self.ch = out_ch//8
        self.conv1 = nn.Sequential(
            ComplexConv(in_ch,  out_ch//8, kernel_size=3, padding=1, stride=1),
            ComplexBatchNorm2d(out_ch // 8),
            ModReLU(out_ch // 8)
        )
        self.conv2 = nn.Sequential(
            ComplexConv(out_ch//8,  out_ch*3//8, kernel_size=3, padding=1, stride=1),
            ComplexBatchNorm2d(out_ch * 3 // 8),
            ModReLU(out_ch * 3 // 8)
        )
        self.conv3 = nn.Sequential(
            ComplexConv(out_ch*3//8,  out_ch//2, kernel_size=3, padding=1, stride=1),
            ComplexBatchNorm2d(out_ch // 2),
            ModReLU(out_ch // 2)
        )
        self.CBAM = CBAM(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.CBAM(out)
        return out

class dilated_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dilated_conv, self).__init__()
        self.ch = out_ch//8
        self.conv1 = nn.Sequential(
            ComplexConv(in_ch,  out_ch//8, kernel_size=3, padding=1, stride=1, dilation=1),
            ComplexBatchNorm2d(out_ch // 8),
            ModReLU(out_ch // 8)
        )
        self.conv2 = nn.Sequential(
            ComplexConv(out_ch//8,  out_ch*3//8, kernel_size=3, padding=2, stride=1, dilation=2),
            ComplexBatchNorm2d(out_ch * 3 // 8),
            ModReLU(out_ch * 3 // 8)
        )
        self.conv3 = nn.Sequential(
            ComplexConv(out_ch*3//8,  out_ch//2, kernel_size=3, padding=3, stride=1, dilation=3),
            ComplexBatchNorm2d(out_ch // 2),
            ModReLU(out_ch // 2)
        )
        self.CBAM = CBAM(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.CBAM(out)
        return out

class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_conv, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv(in_ch,  out_ch, kernel_size=1, padding=0, stride=1),
            ComplexBatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

class MCFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MCFA, self).__init__()
        self.g_conv = general_conv(in_ch, out_ch)
        self.d_conv = dilated_conv(in_ch, out_ch)
        self.r_conv = res_conv(in_ch, out_ch)
        self.relu = ModReLU(out_ch)
    def forward(self, x):
        x1 = self.g_conv(x)
        x2 = self.d_conv(x)
        x3 = self.r_conv(x)
        out = self.relu(x1+x2+x3)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = ComplexConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu1 = ModReLU(out_channels)

        self.conv2 = ComplexConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(out_channels)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                ComplexConv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                ComplexBatchNorm2d(out_channels)
            )

        self.relu2 = ModReLU(out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        skip_out = self.skip_connection(x)

        out += skip_out
        out = self.relu2(out)

        return out

# Attention Gate from Attention U-Net
# Reference: Oktay, O., et al. "Attention U-Net: Learning Where to Look
# for the Pancreas", MIDL 2018.
# Code adapted from: https://github.com/ozan-oktay/Attention-Gated-Networks

class Attention_Gates(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Gates, self).__init__()
        self.W_g = nn.Sequential(
            ComplexConv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ComplexBatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            ComplexConv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            ComplexBatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            ComplexConv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            ComplexBatchNorm2d(1),
            Csigmoid(1)
        )
        self.relu =  ModReLU(F_int)


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.up = ComplexConvTranspose2d(in_ch, in_ch, 2, padding=0, stride=2)
        self.upc1 = ComplexConv(in_ch, out_ch, 3, padding=1)

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.upc1(x1)
        return x1

class CMAUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CMAUNet, self).__init__()
        n = 32
        filters = [n, n * 2, n * 4, n * 8, n * 16]
        self.MCFA1 = MCFA(in_ch, filters[0])
        self.MCFA2 = MCFA(filters[0], filters[1])
        self.MCFA3 = MCFA(filters[1], filters[2])
        self.MCFA4 = MCFA(filters[2], filters[3])
        self.down1 = ComplexConv(filters[0], filters[0], kernel_size=3, stride=2, padding=1)
        self.down2 = ComplexConv(filters[1], filters[1], kernel_size=3, stride=2, padding=1)
        self.down3 = ComplexConv(filters[2], filters[2], kernel_size=3, stride=2, padding=1)
        self.down4 = ComplexConv(filters[3], filters[3], kernel_size=3, stride=2, padding=1)
        self.AG1 = Attention_Gates(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
        self.AG2 = Attention_Gates(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.AG3 = Attention_Gates(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.AG4 = Attention_Gates(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.up1 = up(filters[1],filters[0])
        self.up2 = up(filters[2],filters[1])
        self.up3 = up(filters[3],filters[2])
        self.up4 = up(filters[4],filters[3])
        self.res_conv5 = ResidualBlock(filters[3], filters[4])
        self.res_conv4 = ResidualBlock(filters[4], filters[3])
        self.res_conv3 = ResidualBlock(filters[3], filters[2])
        self.res_conv2 = ResidualBlock(filters[2], filters[1])
        self.res_conv1 = ResidualBlock(filters[1], filters[0])
        self.out_conv = ComplexConv(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.MCFA1(x)
        x2 = self.down1(x1)
        x2 = self.MCFA2(x2)
        x3 = self.down2(x2)
        x3 = self.MCFA3(x3)
        x4 = self.down3(x3)
        x4 = self.MCFA4(x4)
        x5 = self.down4(x4)
        x5 = self.res_conv5(x5)
        u4 = self.up4(x5)
        x4 = self.AG4(g=u4, x=x4)
        u4 = torch.cat([x4, u4], dim=1)
        u4 = self.res_conv4(u4)
        u3 = self.up3(u4)
        x3 = self.AG3(g=u3, x=x3)
        u3 = torch.cat([x3, u3], dim=1)
        u3 = self.res_conv3(u3)
        u2 = self.up2(u3)
        x2 = self.AG2(g=u2, x=x2)
        u2 = torch.cat([x2, u2], dim=1)
        u2 = self.res_conv2(u2)
        u1 = self.up1(u2)
        x1 = self.AG1(g=u1, x=x1)
        u1 = torch.cat([x1, u1], dim=1)
        u1 = self.res_conv1(u1)
        out = self.out_conv(u1)
        return out
