import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.coupling_layer import CouplingLayer
from util import squeeze_2_2


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)

        return x, sldj

    def _pre_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dequantize the input image `x` and convert to logits.
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        sldj = ldj.contiguous().view(ldj.size(0), -1).sum(-1)

        return y, sldj

    def sample(self, size: int, img_size: int = 64):
        """
        Sample from RealNVP model.
        """
        z = torch.randn((size, 3, img_size, img_size), dtype=torch.float32, device='cuda')
        x, _ = self.flows(z, None, True)
        x = torch.sigmoid(x)

        return x


class _RealNVP(nn.Module):
    """
    Recursive builder for a `RealNVP` model.
    """

    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.is_last_block = scale_idx == num_scales - 1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, is_mask_type_wise=False, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, is_mask_type_wise=False, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, is_mask_type_wise=False, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, is_mask_type_wise=False, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, is_mask_type_wise=True,
                              reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, is_mask_type_wise=True, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, is_mask_type_wise=True, reverse_mask=False)
            ])
            self.next_block = _RealNVP(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def _re_squeeze(self, x, sldj, reverse):
        x = squeeze_2_2(x, reverse=False, alt_order=True)
        x, x_split = x.chunk(2, dim=1)
        x, sldj = self.next_block(x, sldj, reverse)
        x = torch.cat((x, x_split), dim=1)
        x = squeeze_2_2(x, reverse=True, alt_order=True)
        return x

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = self._re_squeeze(x, sldj, reverse)
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2_2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2_2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2_2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2_2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = self._re_squeeze(x, sldj, reverse)

        return x, sldj
