import torch

def dirac_orthogonal_(tensor, gain):
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    out_channels = tensor.shape[0]
    in_channels = tensor.shape[1]

    weights = torch.Tensor(out_channels, in_channels)
    torch.nn.init.orthogonal_(weights, gain)

    with torch.no_grad():
        tensor.zero_()
        if dimensions == 3:
            for o in range(out_channels):
                for i in range(in_channels):
                    tensor[o, i, tensor.size(2) // 2] = weights[o, i]
        elif dimensions == 4:
            for o in range(out_channels):
                for i in range(in_channels):
                    tensor[o, i, tensor.size(2) // 2, tensor.size(3) // 2] = weights[o, i]
        else: #dimensions == 5
            for o in range(out_channels):
                for i in range(in_channels):
                    tensor[o, i, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = weights[o, i]
