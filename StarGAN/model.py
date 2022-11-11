import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm.notebook import tqdm
import os
import wandb

from fid import calculate_fid
from layers import CNR2d, ResBlock, DECNR2d
from utils import init_weights, GradientPanaltyLoss, ToNumpy, Denormalize


class Generator(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_channels_ker=64, norm='bnorm', res_blocks_num=6,
                 n_attrs=5):
        super().__init__()
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.num_channels_ker = num_channels_ker
        self.norm = norm
        self.res_blocks_num = res_blocks_num
        self.n_attrs = n_attrs

        self.bias = False if norm == 'bnorm' else True

        self.enc1 = CNR2d(self.num_channels_in, self.num_channels_ker,
                          kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(self.num_channels_ker, 2 * self.num_channels_ker,
                          kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.num_channels_ker, 4 * self.num_channels_ker,
                          kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.res = nn.Sequential(
            *[ResBlock(4 * self.num_channels_ker, 4 * self.num_channels_ker, kernel_size=3, stride=1,
                       padding=1, norm=self.norm, relu=0.0, padding_mode='reflection') for _ in
              range(self.res_blocks_num)])

        self.dec3 = DECNR2d(4 * self.num_channels_ker + self.n_attrs, 2 * self.num_channels_ker,
                            kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.num_channels_ker, self.num_channels_ker,
                            kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(self.num_channels_ker, self.num_channels_out, kernel_size=7,
                          stride=1, padding=3, norm=None, relu=None, bias=False)

    def forward(self, x, y):
        x = torch.cat([x, torch.randn_like(x)], dim=1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        n, _, h, w = x.size()
        y = y.expand((n, self.n_attrs, h, w))
        x = torch.cat([x, y], dim=1)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, num_channels_in=3, num_channels_ker=64, norm='inorm' or None, num_classes=5, nrepeat=6,
                 inp_size=256, out_size=256):
        super().__init__()

        self.num_channels_in = num_channels_in
        self.num_channels_ker = num_channels_ker
        self.norm = norm
        self.num_classes = num_classes
        self.nrepeat = nrepeat

        self.bias = False if norm == 'bnorm' else True

        self.dsc = nn.Sequential(
            CNR2d(1 * self.num_channels_in, self.num_channels_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                  relu=0.1),
            *[CNR2d((2 ** (i - 1)) * self.num_channels_ker, (2 ** i) * self.num_channels_ker,
                    kernel_size=4, stride=2, padding=1, norm='inorm', relu=0.1) for i in range(1, self.nrepeat)],
        )

        self.dsc_src = CNR2d((2 ** (self.nrepeat - 1)) * self.num_channels_ker, 1,
                             kernel_size=4, stride=1, padding=1, norm=None, relu=None, bias=False)
        if num_classes:
            kernel_size = (int(inp_size / (2 ** self.nrepeat)),
                           int(out_size / (2 ** self.nrepeat)))
            self.dsc_cls = CNR2d((2 ** (self.nrepeat - 1)) * self.num_channels_ker, self.num_classes,
                                 kernel_size=kernel_size, stride=1, padding=0, norm=None, relu=None, bias=False)

    def forward(self, x):
        x = self.dsc(x)
        x_src = self.dsc_src(x)
        x_cls = self.dsc_cls(x)
        return x_src, x_cls


class StarGAN:
    def __init__(self, attrs, index2attr, train_continue='on', num_epoch=20, lr_G=2e-4, lr_D=2e-4, batch_size=16,
                 wgt_cls=1.0,
                 wgt_rec=10.0, wgt_gp=10.0, beta1=0.5, _in=64, _out=64, num_channels_ker=64, norm='bnorm',
                 num_freq_disp=100, num_freq_save=1, res_blocks_num=5, n_critic_run=5):
        self.attrs = attrs
        self.index2attr = index2attr
        self.train_continue = train_continue
        self.dir_checkpoint = 'models'
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr_G = lr_G
        self.lr_D = lr_D

        self.wgt_cls = wgt_cls
        self.wgt_rec = wgt_rec
        self.wgt_gp = wgt_gp
        self.beta1 = beta1

        self._in = _in
        self._out = _out
        self.num_channels_ker = num_channels_ker

        self.norm = norm
        self.num_freq_disp = num_freq_disp
        self.num_freq_save = num_freq_save
        self.res_blocks_num = res_blocks_num
        self.len_attrs = len(attrs)
        self.n_critic_run = n_critic_run

        self.transform_inv = transforms.Compose([ToNumpy()])

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def save(self, dir_chck, netGen, netCritic, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)
        torch.save({'netGen': netGen.state_dict(), 'netCritic': netCritic.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   f'{dir_chck}/model_epoch_{epoch}.pth')

    def load(self, dir_chck, netGen, netCritic, optimG=None, optimD=None, mode='train', epoch=1):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load(f'{dir_chck}/model_epoch_{epoch}.pth')

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netGen.load_state_dict(dict_net['netGen'])
            netCritic.load_state_dict(dict_net['netCritic'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])
            return netGen, netCritic, optimG, optimD, epoch
        elif mode == 'test':
            netGen.load_state_dict(dict_net['netGen'])
            return netGen, epoch

    def train(self, loader_train, loader_val, loader_test, epoch=1):
        dir_chck = os.path.join(self.dir_checkpoint)
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])
        num_train = len(loader_train.dataset)
        num_batch_train = int((num_train / self.batch_size) + ((num_train % self.batch_size) != 0))

        netGen = Generator(3 + 3, 3, self.num_channels_ker, self.norm, res_blocks_num=self.res_blocks_num)
        netCritic = Critic(3, self.num_channels_ker, norm='inorm', num_classes=self.len_attrs, inp_size=self._out,
                           out_size=self._out, nrepeat=self.res_blocks_num)
        netGen = init_weights(netGen, init_type='normal', init_gain=0.02)
        netCritic = init_weights(netCritic, init_type='normal', init_gain=0.02)

        loss_REC = nn.L1Loss().to(self.device)  # L1
        loss_SRC = nn.BCEWithLogitsLoss().to(self.device)
        loss_GradPenalty = GradientPanaltyLoss().to(self.device)
        loss_CLS = nn.BCEWithLogitsLoss().to(self.device)  # L1

        optimG = torch.optim.Adam(netGen.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(netCritic.parameters(), lr=self.lr_D, betas=(self.beta1, 0.999))

        st_epoch = 1
        if self.train_continue == 'on':
            netGen, netCritic, optimG, optimD, st_epoch = self.load(dir_chck, netGen, netCritic, optimG, optimD,
                                                                    mode='train', epoch=epoch)
            optimG = torch.optim.Adam(netGen.parameters(), lr=self.lr_G, betas=(self.beta1, 0.999))
            optimD = torch.optim.Adam(netCritic.parameters(), lr=self.lr_D, betas=(self.beta1, 0.999))
        netGen.to(self.device)
        netCritic.to(self.device)
        for epoch in tqdm(range(st_epoch, self.num_epoch + 1), total=self.num_epoch):
            netGen.train()
            netCritic.train()

            for i, data in tqdm(enumerate(loader_train, 1), total=len(loader_train)):
                def show_results(freq):
                    return freq > 0 and (i + 1) % freq == 0

                loss_D_src_train = []
                loss_D_cls_train = []
                loss_D_gp_train = []
                loss_G_src_train = []
                loss_G_cls_train = []
                loss_G_rec_train = []

                input = data[0]
                label_in = data[1].view(-1, self.len_attrs, 1, 1)
                label_out = label_in[torch.randperm(label_in.size(0))]

                input = input.to(self.device)
                label_in = label_in.to(self.device)
                label_out = label_out.to(self.device)

                output = netGen(input, label_in)
                output.to(self.device)
                recon = netGen(output, label_out)

                # backward netCritic
                set_requires_grad(netCritic, True)
                optimD.zero_grad()

                src_in, cls_in = netCritic(input)
                src_out, cls_out = netCritic(output.detach())

                # Calculate Gradient Penalty term
                alpha = torch.rand(input.size(0), 1, 1, 1).to(self.device)
                output_ = (alpha * input + (1 - alpha) * output.detach()).requires_grad_(True)
                src_out_, _ = netCritic(output_)

                # BCE loss
                loss_D_src_in = loss_SRC(src_in, torch.ones_like(src_in))
                loss_D_src_out = loss_SRC(src_out, torch.zeros_like(src_out))

                loss_D_src = 0.5 * (loss_D_src_in + loss_D_src_out)

                cls_in, label_in = self._tensor_to_float(cls_in, label_in)
                loss_D_cls_in = loss_CLS(cls_in, label_in)
                loss_D_cls = loss_D_cls_in

                # Gradient Penalty loss
                loss_D_gp = loss_GradPenalty(src_out_, output_)

                loss_D = loss_D_src + self.wgt_cls * loss_D_cls + self.wgt_gp * loss_D_gp
                loss_D.backward()
                optimD.step()

                loss_D_src_train += [loss_D_src.item()]
                loss_D_cls_train += [loss_D_cls.item()]
                loss_D_gp_train += [loss_D_gp.item()]

                if (i + 1) % self.n_critic_run == 0:
                    # backward netGen
                    set_requires_grad(netCritic, False)
                    optimG.zero_grad()

                    src_out, cls_out = netCritic(output)

                    # BCE Loss
                    loss_G_src = loss_SRC(src_out, torch.ones_like(src_out))

                    cls_out, label_out = self._tensor_to_float(cls_out, label_out)
                    loss_G_cls = loss_CLS(cls_out, label_out)
                    loss_G_rec = loss_REC(input, recon)

                    loss_G = loss_G_src + self.wgt_cls * loss_G_cls + self.wgt_rec * loss_G_rec

                    loss_G.backward()
                    optimG.step()

                    loss_G_src_train += [loss_G_src.item()]
                    loss_G_cls_train += [loss_G_cls.item()]
                    loss_G_rec_train += [loss_G_rec.item()]

                    wandb.log({'loss_D_src': np.mean(loss_D_src_train)})
                    wandb.log({'loss_D_cls': np.mean(loss_D_cls_train)})
                    wandb.log({'loss_D_gp': np.mean(loss_D_gp_train)})
                    wandb.log({'loss_G_src': np.mean(loss_G_src_train)})
                    wandb.log({'loss_G_cls': np.mean(loss_G_cls_train)})
                    wandb.log({'loss_G_rec': np.mean(loss_G_rec_train)})

                if show_results(self.num_freq_disp * 2):
                    input = transform_inv(input)
                    output = transform_inv(output)
                    recon = transform_inv(recon)

                    inp_attrs = set(
                        [self.index2attr[i] for i, att_val in enumerate(label_in[0].detach().view(-1)) if att_val == 1])
                    out_attrs = set([self.index2attr[i] for i, att_val in enumerate(label_out[0].detach().view(-1)) if
                                     att_val == 1])
                    ep = num_batch_train * (epoch - 1) + i
                    wandb.log({f'iter {ep}': [
                        wandb.Image(input[0], caption=f"input new attrs {' '.join(list(inp_attrs - out_attrs))}"),
                        wandb.Image(output[0],
                                    caption=f"output new attrs {' '.join(list(out_attrs - inp_attrs))}"),
                        wandb.Image(recon[0], caption='recon')]})

            if (epoch % self.num_freq_save) == 0:
                self.save(dir_chck, netGen, netCritic, optimG, optimD, epoch)

            fid = self.calc_fid(loader_val, epoch)
            wandb.log({'fid': fid})
            self.test(loader_test, epoch)

    def _tensor_to_float(self, inp, out):
        inp = inp.to('cpu')
        out = out.to('cpu')
        out = out.float()
        inp = torch.FloatTensor(inp)
        out = torch.FloatTensor(out)
        return inp, out

    def test(self, loader_test, epoch):
        dir_chck = os.path.join(self.dir_checkpoint)
        num_test = len(loader_test.dataset)

        netGen = Generator(3 + 3, 3, self.num_channels_ker, self.norm,
                           res_blocks_num=self.res_blocks_num)
        netCritic = Critic(3, self.num_channels_ker, norm='inorm', num_classes=self.len_attrs, inp_size=self._out,
                           out_size=self._out)

        netGen, st_epoch = self.load(dir_chck, netGen, netCritic, mode='test', epoch=epoch)
        netGen.to(self.device)
        netCritic.to(self.device)
        with torch.no_grad():
            netGen.eval()
            for i, data in tqdm(enumerate(loader_test, 1), total=num_test):
                images = []
                is_input = True
                for j, attr in enumerate(self.attrs):
                    if attr not in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Mustache', 'Young', 'Bald', 'Smiling']:
                        continue
                    input = data[0]
                    label_in = data[1].view(-1, self.len_attrs, 1, 1)
                    label_out = torch.zeros_like(label_in)
                    label_out[:, j, :, :] = 1

                    input = input.to(self.device)
                    label_out = label_out.to(self.device)

                    output = netGen(input, label_out)
                    input = self.transform_inv(input)
                    output = self.transform_inv(output)
                    if is_input:
                        images.append(wandb.Image(input.squeeze(), caption='input'))
                        is_input = False
                    images.append(wandb.Image(output.squeeze(), caption=f'output_{attr}'))

                wandb.log({f'test iter {i}': images})

    @torch.no_grad()
    def calc_fid(self, loader, epoch=12):
        dir_chck = os.path.join(self.dir_checkpoint)
        netGen = Generator(3 + 3, 3, self.num_channels_ker, self.norm,
                           res_blocks_num=self.res_blocks_num)
        netCritic = Critic(3, self.num_channels_ker, norm='inorm', num_classes=self.len_attrs, inp_size=self._out,
                           out_size=self._out)

        netGen, _ = self.load(dir_chck, netGen, netCritic, mode='test', epoch=epoch)
        netGen.to(self.device)

        fid = calculate_fid(netGen, loader, attrs_len=self.len_attrs)
        return fid


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
