from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dymn.utils import make_divisible, cnn_out_size


class DynamicInvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_dy_block: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class DynamicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 context_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=0,
                 groups=1,
                 att_groups=1,
                 bias=False,
                 k=4,
                 temp_schedule=(30, 1, 1, 0.05)
                 ):
        super(DynamicConv, self).__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        # att_groups splits the channels into 'att_groups' groups and predicts separate attention weights
        # for each of the groups; did only give slight improvements in our experiments and not mentioned in paper
        self.att_groups = att_groups

        # Equation 6 in paper: obtain coefficients for K attention weights over conv. kernels
        self.residuals = nn.Sequential(
                nn.Linear(context_dim, k * self.att_groups)
        )

        # k sets of weights for convolution
        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)

        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_channels), requires_grad=True)
        else:
            self.bias = None

        self._initialize_weights(weight, self.bias)

        weight = weight.view(1, k, att_groups, out_channels,
                             in_channels // groups, kernel_size, kernel_size)

        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def _initialize_weights(self, weight, bias):
        init_func = partial(nn.init.kaiming_normal_, mode="fan_out")
        for i in range(self.k):
            init_func(weight[i])
            if bias is not None:
                nn.init.zeros_(bias[i])

    def forward(self, x, g=None):
        b, c, f, t = x.size()
        g_c = g[0].view(b, -1)
        residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
        attention = F.softmax(residuals / self.temperature, dim=-1)

        # attention shape: batch_size x 1 x 1 x k
        # self.weight shape: 1 x 1 x k x out_channels * (in_channels // groups) * kernel_size ** 2
        aggregate_weight = (attention @ self.weight).transpose(1, 2).reshape(b, self.out_channels,
                                                                             self.in_channels // self.groups,
                                                                             self.kernel_size, self.kernel_size)

        # aggregate_weight shape: batch_size x out_channels x in_channels // groups x kernel_size x kernel_size
        aggregate_weight = aggregate_weight.view(b * self.out_channels, self.in_channels // self.groups,
                                                 self.kernel_size, self.kernel_size)
        # each sample in the batch has different weights for the convolution - therefore batch and channel dims need to
        # be merged together in channel dimension
        x = x.view(1, -1, f, t)
        if self.bias is not None:
            aggregate_bias = torch.mm(attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * b)

        # output shape: 1 x batch_size * channels x f_bands x time_frames
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        return output

    def update_params(self, epoch):
        # temperature schedule for attention weights
        # see Equation 5: tau = temperature
        t0 = self.T_max - self.T0_slope * epoch
        t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
        self.temperature = max(t0, t1, self.T_min)
        print(f"Setting temperature for attention over kernels to {self.temperature}")


class DyReLU(nn.Module):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.M = M

        self.coef_net = nn.Sequential(
                nn.Linear(context_dim, 2 * M)
        )

        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * M + [0.5] * M).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * M - 1)).float())

    def get_relu_coefs(self, x):
        theta = self.coef_net(x)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, g):
        raise NotImplementedError


class DyReLUB(DyReLU):
    def __init__(self, channels, context_dim, M=2):
        super(DyReLUB, self).__init__(channels, context_dim, M)
        # Equation 4 in paper: obtain coefficients for M linear mappings for each of the C channels
        self.coef_net[-1] = nn.Linear(context_dim, 2 * M * self.channels)

    def forward(self, x, g):
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.size()
        h_c = g[0].view(b, -1)
        theta = self.get_relu_coefs(h_c)

        relu_coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        # relu_coefs shape: batch_size x channels x 1 x 1 x 2*M
        # x shape: batch_size x channels x f_bands x time_frames
        x_mapped = x.unsqueeze(-1) * relu_coefs[:, :, :, :, :self.M] + relu_coefs[:, :, :, :, self.M:]
        if self.M == 2:
            # torch.maximum turned out to be faster than torch.max for M=2
            result = torch.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = torch.max(x_mapped, dim=-1)[0]
        return result


class CoordAtt(nn.Module):
    def __init__(self):
        super(CoordAtt, self).__init__()

    def forward(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = g_cf.sigmoid()
        a_t = g_ct.sigmoid()
        # recalibration with channel-frequency and channel-time weights
        out = x * a_f * a_t
        return out


class DynamicWrapper(torch.nn.Module):
    # wrap a pytorch module in a dynamic module
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, g):
        return self.module(x)


class ContextGen(nn.Module):
    def __init__(self, context_dim, in_ch, exp_ch, norm_layer, stride: int = 1):
        super(ContextGen, self).__init__()

        # shared linear layer implemented as a 2D convolution with 1x1 kernel
        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)

        # separate linear layers for Coordinate Attention
        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if stride > 1:
            # sequence pooling for Coordinate Attention
            self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        else:
            self.pool_f = nn.Sequential()
            self.pool_t = nn.Sequential()

    def forward(self, x, g):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)

        g_cat = torch.cat([cf, ct], dim=2)
        # joint frequency and time sequence transformation (S_F and S_T in the paper)
        g_cat = self.joint_norm(self.joint_conv(g_cat))
        g_cat = self.joint_act(g_cat)

        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        # pooling over sequence dimension to get context vector of size H to parameterize Dy-ReLU and Dy-Conv
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        g_cf, g_ct = self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct))

        # g[0]: context vector of size H to parameterize Dy-ReLU and Dy-Conv
        # g[1], g[2]: frequency and time sequences for Coordinate Attention
        g = (h_c, g_cf, g_ct)
        return g


class DY_Block(nn.Module):
    def __init__(
            self,
            cnf: DynamicInvertedResidualConfig,
            context_ratio: int = 4,
            max_context_size: int = 128,
            min_context_size: int = 32,
            temp_schedule: tuple = (30, 1, 1, 0.05),
            dyrelu_k: int = 2,
            dyconv_k: int = 4,
            no_dyrelu: bool = False,
            no_dyconv: bool = False,
            no_ca: bool = False,
            **kwargs: Any
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        # context_dim is denoted as 'H' in the paper
        self.context_dim = np.clip(make_divisible(cnf.expanded_channels // context_ratio, 8),
                                   make_divisible(min_context_size * cnf.width_mult, 8),
                                   make_divisible(max_context_size * cnf.width_mult, 8)
                                   )

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            if no_dyconv:
                self.exp_conv = DynamicWrapper(
                    nn.Conv2d(
                        cnf.input_channels,
                        cnf.expanded_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        dilation=(1, 1),
                        padding=0,
                        bias=False
                    )
                )
            else:
                self.exp_conv = DynamicConv(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    self.context_dim,
                    kernel_size=1,
                    k=dyconv_k,
                    temp_schedule=temp_schedule,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=False
                )

            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = DynamicWrapper(activation_layer(inplace=True))
        else:
            self.exp_conv = DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = DynamicWrapper(nn.Identity())

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        padding = (cnf.kernel - 1) // 2 * cnf.dilation
        if no_dyconv:
            self.depth_conv = DynamicWrapper(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.expanded_channels,
                    kernel_size=(cnf.kernel, cnf.kernel),
                    groups=cnf.expanded_channels,
                    stride=(stride, stride),
                    dilation=(cnf.dilation, cnf.dilation),
                    padding=padding,
                    bias=False
                )
            )
        else:
            self.depth_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.expanded_channels,
                self.context_dim,
                kernel_size=cnf.kernel,
                k=dyconv_k,
                temp_schedule=temp_schedule,
                groups=cnf.expanded_channels,
                stride=stride,
                dilation=cnf.dilation,
                padding=padding,
                bias=False
            )
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = DynamicWrapper(activation_layer(inplace=True)) if no_dyrelu \
            else DyReLUB(cnf.expanded_channels, self.context_dim, M=dyrelu_k)

        self.ca = DynamicWrapper(nn.Identity()) if no_ca else CoordAtt()

        # project
        if no_dyconv:
            self.proj_conv = DynamicWrapper(
                nn.Conv2d(
                    cnf.expanded_channels,
                    cnf.out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    dilation=(1, 1),
                    padding=0,
                    bias=False
                )
            )
        else:
            self.proj_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.out_channels,
                self.context_dim,
                kernel_size=1,
                k=dyconv_k,
                temp_schedule=temp_schedule,
                stride=1,
                dilation=1,
                padding=0,
                bias=False,
            )

        self.proj_norm = norm_layer(cnf.out_channels)

        context_norm_layer = norm_layer
        self.context_gen = ContextGen(self.context_dim, cnf.input_channels, cnf.expanded_channels,
                                      norm_layer=context_norm_layer, stride=stride)

    def forward(self, x, g=None):
        # x: CNN feature map (C x F x T)
        inp = x

        g = self.context_gen(x, g)
        x = self.exp_conv(x, g)
        x = self.exp_norm(x)
        x = self.exp_act(x, g)

        x = self.depth_conv(x, g)
        x = self.depth_norm(x)
        x = self.depth_act(x, g)
        x = self.ca(x, g)

        x = self.proj_conv(x, g)
        x = self.proj_norm(x)

        if self.use_res_connect:
            x += inp
        return x
