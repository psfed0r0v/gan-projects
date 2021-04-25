import torch
import torch.nn as nn

from models.resnet import ResNet
from util import checkerboard_mask


class CouplingLayer(nn.Module):
    """
    Coupling layer in RealNVP.
    """

    def __init__(self, in_channels: int, mid_channels: int, num_blocks: int, is_mask_type_wise: bool, reverse_mask: bool):
        super().__init__()

        # Save mask info
        self.is_mask_type_wise = is_mask_type_wise
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if not self.is_mask_type_wise:
            in_channels //= 2
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1,
                             double_after_norm=(not self.is_mask_type_wise))

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def _check_nans(self, t):
        if torch.isnan(t).any():
            raise RuntimeError('Scale factor has NaN entries')

    def forward(self, x, sldj=None, reverse=True):
        if not self.is_mask_type_wise:
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                self._check_nans(inv_exp_s)
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                self._check_nans(exp_s)
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                self._check_nans(inv_exp_s)
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                self._check_nans(exp_s)
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class Rescale(nn.Module):
    """
    Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
