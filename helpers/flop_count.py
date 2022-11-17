import torch
import torch.nn as nn


# adapted from PANNs (https://github.com/qiuqiangkong/audioset_tagging_cnn)

def count_macs(model, spec_size):
    list_conv2d = []

    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        assert batch_size == 1
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # overall macs count is:
        # kernel**2 * in_channels/groups * out_channels * out_width * out_height
        macs = batch_size * params * output_height * output_width

        list_conv2d.append(macs)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        assert batch_size == 1
        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement()

        # overall macs count is equal to the number of parameters in layer
        macs = batch_size * (weight_ops + bias_ops)
        list_linear.append(macs)

    def foo(net):
        if net.__class__.__name__ == 'Conv2dStaticSamePadding':
            net.register_forward_hook(conv2d_hook)
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)
    with torch.no_grad():
        model(input)

    total_macs = sum(list_conv2d) + sum(list_linear)

    print("*************Computational Complexity (multiply-adds) **************")
    print("Number of Convolutional Layers: ", len(list_conv2d))
    print("Number of Linear Layers: ", len(list_linear))
    print("Relative Share of Convolutional Layers: {:.2f}".format((sum(list_conv2d) / total_macs)))
    print("Relative Share of Linear Layers: {:.2f}".format(sum(list_linear) / total_macs))
    print("Total MACs (multiply-accumulate operations in Billions): {:.2f}".format(total_macs/10**9))
    print("********************************************************************")
    return total_macs


def count_macs_transformer(model, spec_size):
    """Count macs. Code modified from others' implementation.
        """
    list_conv2d = []

    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        assert batch_size == 1
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        # overall macs count is:
        # kernel**2 * in_channels/groups * out_channels * out_width * out_height
        macs = batch_size * params * output_height * output_width

        list_conv2d.append(macs)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() >= 2 else 1
        assert batch_size == 1
        if input[0].dim() == 3:
            # (batch size, sequence length, embeddings size)
            batch_size, seq_len, embed_size = input[0].size()

            weight_ops = self.weight.nelement()
            bias_ops = self.bias.nelement() if self.bias is not None else 0
            # linear layer applied position-wise, multiply with sequence length
            macs = batch_size * (weight_ops + bias_ops) * seq_len
        else:
            # classification head
            # (batch size, embeddings size)
            batch_size, embed_size = input[0].size()
            weight_ops = self.weight.nelement()
            bias_ops = self.bias.nelement() if self.bias is not None else 0
            # overall macs count is equal to the number of parameters in layer
            macs = batch_size * (weight_ops + bias_ops)
        list_linear.append(macs)

    list_att = []

    def attention_hook(self, input, output):
        # here we only calculate the attention macs; linear layers are processed in linear_hook
        batch_size, seq_len, embed_size = input[0].size()

        # 2 times embed_size * seq_len**2
        # - computing the attention matrix: embed_size * seq_len**2
        # - multiply attention matrix with value matrix: embed_size * seq_len**2
        macs = batch_size * embed_size * seq_len * seq_len * 2
        list_att.append(macs)

    def foo(net):
        childrens = list(net.children())
        if net.__class__.__name__ == "MultiHeadAttention":
            net.register_forward_hook(attention_hook)
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)

    with torch.no_grad():
        model(input)

    total_macs = sum(list_conv2d) + sum(list_linear) + sum(list_att)

    print("*************Computational Complexity (multiply-adds) **************")
    print("Number of Convolutional Layers: ", len(list_conv2d))
    print("Number of Linear Layers: ", len(list_linear))
    print("Number of Attention Layers: ", len(list_att))
    print("Relative Share of Convolutional Layers: {:.2f}".format((sum(list_conv2d) / total_macs)))
    print("Relative Share of Linear Layers: {:.2f}".format(sum(list_linear) / total_macs))
    print("Relative Share of Attention Layers: {:.2f}".format(sum(list_att) / total_macs))
    print("Total MACs (multiply-accumulate operations in Billions): {:.2f}".format(total_macs/10**9))
    print("********************************************************************")
    return total_macs
