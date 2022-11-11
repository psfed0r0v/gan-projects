import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """
    Get the NLL loss for a RealNVP model.
    """

    def __init__(self, k=256):
        super().__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
