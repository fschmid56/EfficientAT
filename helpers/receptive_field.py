import torch
import torch.nn as nn


def receptive_field_cnn(model, spec_size):
    kernel_sizes = []
    strides = []

    def conv2d_hook(self, input, output):
        kernel_sizes.append(self.kernel_size[0])
        strides.append(self.stride[0])

    def foo(net):
        if net.__class__.__name__ == 'Conv2d':
            net.register_forward_hook(conv2d_hook)
        childrens = list(net.children())
        if isinstance(net, nn.Conv2d):
            net.register_forward_hook(conv2d_hook)
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)
    with torch.no_grad():
        model(input)

    r = 1
    for k, s in zip(kernel_sizes[::-1], strides[::-1]):
        r = s * r + (k - s)

    return r
