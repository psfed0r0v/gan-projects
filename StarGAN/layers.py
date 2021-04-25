import torch
import torch.nn as nn
import torch.nn.functional as F


class CNR2d(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0,
                 drop=None,
                 bias=None):
        super().__init__()
        if not bias:
            bias = False if norm == 'bnorm' else True

        layers = []
        layers += [nn.Conv2d(num_channels_in, num_channels_out, kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if norm:
            layers += [Norm2d(num_channels_out, norm)]

        if relu:
            layers += [ReLU(relu)]

        if drop:
            layers += [nn.Dropout2d(drop)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class DECNR2d(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, kernel_size=4, stride=1, padding=1, output_padding=0,
                 norm='bnorm', relu=0.0,
                 drop=None, bias=None):
        super().__init__()
        if not bias:
            bias = False if norm == 'bnorm' else True

        layers = []
        layers += [Deconv2d(num_channels_in, num_channels_out, kernel_size=kernel_size, stride=stride, padding=padding,
                            output_padding=output_padding, bias=bias)]

        if norm:
            layers += [Norm2d(num_channels_out, norm)]

        if relu:
            layers += [ReLU(relu)]

        if drop:
            layers += [nn.Dropout2d(drop)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection',
                 norm='inorm',
                 relu=0.0, bias=None):
        super().__init__()
        if not bias:
            bias = False if norm == 'bnorm' else True

        self.block = nn.Sequential(
            Padding(padding, padding_mode=padding_mode),
            CNR2d(num_channels_in, num_channels_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm,
                  relu=relu, bias=bias),

            Padding(padding, padding_mode=padding_mode),
            CNR2d(num_channels_in, num_channels_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm,
                  relu=None, bias=bias)
        )

    def forward(self, x):
        return x + self.block(x)


class Deconv2d(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, kernel_size=4, stride=1, padding=1, output_padding=0,
                 bias=True):
        super().__init__()
        # self.block = nn.ConvTranspose2d(num_channels_in, num_channels_out, kernel_size=kernel_size, stride=stride, padding=padding,
        #                                 output_padding=output_padding, bias=bias)
        #

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super().__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super().__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super().__init__()
        if padding_mode == 'reflection':
            self.padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)
