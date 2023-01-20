import torch


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
    def first_inv_residual_block_hook(self, input, output):
        inv_residual_elems.append(global_in_elements[0] + output[0].nelement())

    def inv_residual_hook(self, input, output):
        mem = input[0].nelement() + output[0].nelement()
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
            if len(net.block) > 3:
                # contains Squeeze-and-Excitation Layer -> cannot use memory efficient inference
                # -> must fully materialize all convs in block
                # -> last conv layer has max sum of input and output activation sizes
                net.block[3].register_forward_hook(inv_residual_hook)
            elif len(net.block) == 3:
                # block with no Squeeze-and-Excitation
                # can use memory efficient inference, no need to fully materialize expanded channel representation
                net.register_forward_hook(inv_residual_hook)
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
