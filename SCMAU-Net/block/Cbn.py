#  Complex batch normalization, codes were referenced from:
#  from Wei-CheChen, etc. Voice Separation Using Deep Complex Network
#  https://github.com/ChihebTrabelsi/deep_complex_networks'
#  https: // github.com / fchollet / keras / blob / master / keras / layers / normalization.py
# 'https://github.com/wavefrontshaping/complexPyTorch'

import torch
from torch.nn import Module, Parameter, init
import torch.nn as nn
import math
import sys
from torch.autograd import Variable

class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, x):
        ch = x.shape[1]
        input_r = x[:,:ch//2,:,:]
        input_i = x[:,ch//2:,:,:]
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum


        if self.training:

            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])


            mean = torch.stack((mean_r,mean_i),dim=1)

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r-mean_r[None, :, None, None]
            input_i = input_i-mean_i[None, :, None, None]

            n = input_r.numel() / input_r.size(1)
            Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

            input_r = input_r-mean[None,:,0,None,None]
            input_i = input_i-mean[None,:,1,None,None]

        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
                           Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]


        return torch.cat([input_r, input_i],dim=1)



class ComplexBatchNormalization(Module):
    def __init__(self,
                 channel_dim,
                 epsilon=1e-5,
                 momentum=0.9
                 ):
        super(ComplexBatchNormalization, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.channel_dim = channel_dim
        self.initialize_parameters()

    def initialize_parameters(self):

        self.Vrr_moving = Variable(torch.cuda.FloatTensor([1 / math.sqrt(2)] * (self.channel_dim)),
                                   requires_grad=False)
        self.Vri_moving = Variable(torch.cuda.FloatTensor([0] * (self.channel_dim)), requires_grad=False)
        self.Vii_moving = Variable(torch.cuda.FloatTensor([1 / math.sqrt(2)] * (self.channel_dim)), requires_grad=False)

        self.moving_mean = Variable(torch.cuda.FloatTensor([0] * (self.channel_dim * 2)), requires_grad=False)

        self.Vrr_moving = self.Vrr_moving.view(self.Vri_moving.size(0), 1, 1, 1)
        self.Vri_moving = self.Vri_moving.view(self.Vri_moving.size(0), 1, 1, 1)
        self.Vii_moving = self.Vii_moving.view(self.Vii_moving.size(0), 1, 1, 1)
        self.moving_mean = self.moving_mean.view(self.moving_mean.size(0), 1, 1, 1)

        self.gamma_rr = nn.Parameter(
            torch.cuda.FloatTensor([1 / math.sqrt(2)] * self.channel_dim))
        self.gamma_ri = nn.Parameter(torch.cuda.FloatTensor([0] * self.channel_dim))
        self.gamma_ii = nn.Parameter(torch.cuda.FloatTensor([1 / math.sqrt(2)] * self.channel_dim))
        self.beta = nn.Parameter(torch.cuda.FloatTensor([0] * (self.channel_dim * 2)))

    def moving_mean_update(self, moving_mean, mu, momentum):
        return (moving_mean * momentum + mu * (1. - momentum))

    def complex_standardization(self, input_centred, Vrr, Vii, Vri):
        input_shape = input_centred.size()

        ndim = len(input_shape)
        channel_dim = input_shape[0] // 2

        tau = Vrr + Vii
        delta = (Vrr * Vii) - Vri ** 2

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        inverse_st = 1.0 / (s * t)
        Wrr = ((Vii + s) * inverse_st).view(channel_dim, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).view(channel_dim, 1, 1, 1)
        Wri = (-Vri * inverse_st).view(channel_dim, 1, 1, 1)

        W_cat_real = torch.cat([Wrr, Wii], dim=0)
        W_cat_imag = torch.cat([Wri, Wri], dim=0)

        if ndim == 4:
            centred_real = input_centred[:channel_dim, :, :, :]
            centred_imag = input_centred[channel_dim:, :, :, :]
        else:
            sys.exit('Sorry! Have not handled the case that input_dim != 3')

        rolled_input = torch.cat([centred_imag, centred_real], dim=0)

        output = W_cat_real * input_centred + W_cat_imag * rolled_input
        return output

    def ComplexBN(self, input_centred, Vrr, Vii, Vri,
                  beta,
                  gamma_rr, gamma_ri, gamma_ii):
        input_shape = input_centred.size()
        ndim = len(input_shape)
        channel_dim = input_shape[0] // 2

        standardized_output = self.complex_standardization(input_centred, Vrr, Vii, Vri)

        broadcast_gamma_rr = gamma_rr.view(channel_dim, 1, 1, 1)
        broadcast_gamma_ri = gamma_ri.view(channel_dim, 1, 1, 1)
        broadcast_gamma_ii = gamma_ii.view(channel_dim, 1, 1, 1)

        gamma_cat_real = torch.cat([broadcast_gamma_rr, broadcast_gamma_ii], dim=0)
        gamma_cat_imag = torch.cat([broadcast_gamma_ri, broadcast_gamma_ri], dim=0)

        if ndim == 4:
            centred_real = standardized_output[:channel_dim, :, :, :]
            centred_imag = standardized_output[channel_dim:, :, :, :]
        else:
            sys.exit('Sorry! Have not handled the case that input_dim != 3')

        rolled_standardized_output = torch.cat([centred_imag, centred_real], dim=0)

        broadcast_beta = beta.view(channel_dim * 2, 1, 1, 1)

        returned = gamma_cat_real * standardized_output + gamma_cat_imag * rolled_standardized_output + broadcast_beta
        return returned

    def forward(self, x):
        input_shape = x.size()

        ndim = len(input_shape)
        channel_dim = input_shape[1] // 2

        x_permute = x.permute(1, 0, 2, 3).contiguous()

        if (self.training == False):
            inference_centred = x_permute - self.moving_mean
            returned = self.ComplexBN(inference_centred, self.Vrr_moving, self.Vii_moving,
                                      self.Vri_moving, self.beta, self.gamma_rr, self.gamma_ri,
                                      self.gamma_ii)
            returned = returned.permute(1, 0, 2, 3).contiguous()
            return returned
        else:
            x_reshape = x_permute.view(channel_dim * 2, -1)
            mu = torch.mean(x_reshape, dim=1).view(channel_dim * 2, 1, 1, 1)
            input_centred = x_permute - mu
            centred_squared = input_centred ** 2

            if ndim == 4:
                centred_squared_real = centred_squared[:channel_dim, :, :, :]
                centred_squared_imag = centred_squared[channel_dim:, :, :, :]
                centred_real = input_centred[:channel_dim, :, :, :]
                centred_imag = input_centred[channel_dim:, :, :, :]
            else:
                sys.exit('Sorry! Have not handled the case that input_dim != 3')

            Vrr = (torch.mean(centred_squared_real.view(channel_dim, -1), dim=1) + self.epsilon).view(channel_dim, 1, 1,
                                                                                                      1)
            Vii = (torch.mean(centred_squared_imag.view(channel_dim, -1), dim=1) + self.epsilon).view(channel_dim, 1, 1,
                                                                                                      1)
            Vri = (torch.mean((centred_real * centred_imag).view(channel_dim, -1), dim=1)).view(channel_dim, 1, 1,
                                                                                                1)

            self.moving_mean = self.moving_mean_update(self.moving_mean, mu, self.momentum)
            self.Vrr_moving = self.moving_mean_update(self.Vrr_moving, Vrr, self.momentum)
            self.Vri_moving = self.moving_mean_update(self.Vri_moving, Vri, self.momentum)
            self.Vii_moving = self.moving_mean_update(self.Vii_moving, Vii, self.momentum)

            input_bn = self.ComplexBN(input_centred, Vrr, Vii, Vri,
                                      self.beta,
                                      self.gamma_rr, self.gamma_ri, self.gamma_ii)  # 32 x 20 x 5

            input_bn = input_bn.permute(1, 0, 2, 3)

            return input_bn