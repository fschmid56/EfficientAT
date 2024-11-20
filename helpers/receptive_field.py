import torch
import torch.nn as nn
from models.dymn.dy_block import DynamicConv


def get_values(x):
    return (x, x) if isinstance(x, int) else x


def receptive_field_cnn(model, spec_size):
    kernel_sizes = []
    strides = []
    dilation = []

    def conv2d_hook(obj, input, output):
        kernel_sizes.append(get_values(obj.kernel_size))
        strides.append(get_values(obj.stride))
        dilation.append(get_values(obj.dilation))

    def foo(net):
        if isinstance(net, nn.Conv2d) or isinstance(net, DynamicConv):
            net.register_forward_hook(conv2d_hook)
        children = list(net.children())
        for c in children:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)
    with torch.no_grad():
        model(input)

    rf_freq = 1
    rf_time = 1
    for k, s, d in zip(kernel_sizes[::-1], strides[::-1], dilation[::-1]):
        effective_k0 = (k[0] - 1) * d[0] + 1
        effective_k1 = (k[1] - 1) * d[1] + 1
        rf_freq = s[0] * rf_freq + (effective_k0 - s[0])
        rf_time = s[1] * rf_time + (effective_k1 - s[1])

    return rf_freq, rf_time
