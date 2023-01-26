import torch
import torch.nn as nn


# analytical memory profiling as in:
#   https://proceedings.neurips.cc/paper/2021/file/1371bccec2447b5aa6d96d2a540fb401-Paper.pdf
#   "memory required for a layer is the sum of input and output activation
#           (since weights can be partially fetched from Flash"
# calculated using optimization described in https://arxiv.org/pdf/1801.04381.pdf (memory efficient inference)

def peak_memory_mnv3(model, spec_size, bits_per_elem=16):
    global_in_elements = []
    def in_conv_hook(self, input, output):
        global_in_elements.append(input[0].nelement())

    inv_residual_elems = []
    def first_inv_residual_block_hook(self, input, output, slice=8):
        mem = global_in_elements[-1] + output[0].nelement()
        # we need to only partially materialize internal block representation, we assume 8 parallel path per default
        block_in_t = input[0].size(3)
        block_in_f = input[0].size(2)
        ch = input[0].size(1)
        mem += block_in_t * block_in_f * ch / slice  # repr. before depth-wise
        mem += block_in_t * block_in_f * ch / slice  # repr. after depth-wise
        inv_residual_elems.append(mem)

    res_elements = []
    def res_hook(self, input, output):
        res_elements.append(output[0].nelement())

    def inv_residual_hook(self, input, output):
        mem = input[0].nelement() + output[0].nelement()
        # add possible memory for residual connection
        mem += res_elements[-1]
        inv_residual_elems.append(mem)

    def inv_no_residual_hook(self, input, output, slice=8):
        mem = input[0].nelement() + output[0].nelement()
        # we need to only partially materialize internal block representation, we assume 8 parallel path per default
        block_in_t = input[0].size(3)
        block_in_f = input[0].size(2)
        stride = self.block[1][0].stride[0]
        mem += block_in_t * block_in_f * self.block[0].out_channels / slice  # repr. before depth-wise
        next_in_f = block_in_f // stride
        next_in_t = block_in_t // stride
        mem += next_in_t * next_in_f * self.block[0].out_channels / slice  # repr. after depth-wise
        inv_residual_elems.append(mem)

    def foo(net):
        children = []
        if hasattr(net, "features"):
            # first call to foo with full network
            # treat first ConvNormActivation and InvertedResidual - can be calculated memory efficient
            net.features[0].register_forward_hook(in_conv_hook)
            net.features[1].register_forward_hook(first_inv_residual_block_hook)
            children = list(net.features.children())[2:]
        elif net.__class__.__name__ == 'InvertedResidual':
            # account for residual connection if Squeeze-and-Excitation block
            net.block.register_forward_hook(res_hook)

            if len(net.block) > 3:
                # contains Squeeze-and-Excitation Layer -> cannot use memory efficient inference
                # -> must fully materialize all convs in block
                # -> last conv layer has max sum of input and output activation sizes
                net.block[3].register_forward_hook(inv_residual_hook)
            elif len(net.block) == 3:
                # block with no Squeeze-and-Excitation
                # can use memory efficient inference, no need to fully materialize expanded channel representation
                net.register_forward_hook(inv_no_residual_hook)
            else:
                raise ValueError("Can treat only MobileNetV3 blocks. Block 1 consists of 2 modules and following"
                                 "blocks of 3 or 4 modules. Block 1 must be treated differently.")
        else:
            children = list(net.children())

        for c in children:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)
    with torch.no_grad():
        model(input)

    block_mems = [elem * bits_per_elem / (8 * 1000) for elem in inv_residual_elems]
    peak_mem = max(block_mems)

    print("*************Memory Complexity (kB) **************")
    for i, block_mem in enumerate(block_mems):
        print(f"block {i + 1} memory: {block_mem} kB")
    print("**************************************************")
    print("Analytical peak memory: ", peak_mem, " kB")
    print("**************************************************")
    return peak_mem


def peak_memory_cnn(model, spec_size, bits_per_elem=16):
    first_conv_in_block = [True]
    res_elems = []  # initialized with one 0 for input conv

    def res_hook(self, input, output):
        first_conv_in_block[0] = True
        res_elems.append(output[0].nelement())

    conv_activation_elems = []

    def conv2d_res_hook(self, input, output):
        mem = input[0].nelement() + output[0].nelement()
        # maybe have to add size of parallel residual path
        if not first_conv_in_block[0]:
            mem += res_elems[-1]
        else:
            first_conv_in_block[0] = False
        conv_activation_elems.append(mem)

    def conv2d_hook(self, input, output):
        mem = input[0].nelement() + output[0].nelement()
        conv_activation_elems.append(mem)

    def foo(net, residual_block=False):
        if hasattr(net, "features"):
            net.features[0].register_forward_hook(res_hook)
        if net.__class__.__name__ == 'InvertedResidual':
            net.register_forward_hook(res_hook)
            if net.use_res_connect:
                residual_block = True
        if isinstance(net, nn.Conv2d):
            if residual_block:
                net.register_forward_hook(conv2d_res_hook)
            else:
                net.register_forward_hook(conv2d_hook)

        for c in net.children():
            foo(c, residual_block)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    input = torch.rand(spec_size).to(device)
    with torch.no_grad():
        model(input)

    conv_act_mems = [elem * bits_per_elem / (8 * 1000) for elem in conv_activation_elems]
    peak_mem = max(conv_act_mems)

    print("*************Memory Complexity (kB) **************")
    for i, conv_mem in enumerate(conv_act_mems):
        print(f"conv {i + 1} memory: {conv_mem} kB")
    print("**************************************************")
    print("Analytical peak memory: ", peak_mem, " kB")
    print("**************************************************")
    return peak_mem
