import torch


def mean_dim(tensor, dim=None, keepdims=False):
    """
    Take the mean along multiple dimensions.
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor
