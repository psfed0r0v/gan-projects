import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm.notebook import tqdm
import os
import wandb
import torch.nn.functional as F

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
            *[ResBlock(4 * self.num_channels_ker + 1, 4 * self.num_channels_ker + 1, kernel_size=3, stride=1,
                       padding=1, norm=self.norm, relu=0.0, padding_mode='reflection') for _ in
              range(self.res_blocks_num)])

        self.dec3 = DECNR2d(4 * self.num_channels_ker + 1, 2 * self.num_channels_ker,
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

        y = torch.cat((y, torch.zeros(y.shape[0], -y.shape[1] + x.shape[-1] * x.shape[-2]).to('cuda')), -1)
        y = y.reshape(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, y], dim=1)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, num_channels_in=3, num_channels_ker=64, norm='inorm' or None, num_classes=5, nrepeat=6,
                 inp_size=256, out_size=256):
        super().__init__()

        self.norm = norm
        self.num_classes = num_classes

        self.bias = False if norm == 'bnorm' else True

        self.main = nn.Sequential(
            nn.Conv2d(num_channels_in, num_channels_ker, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_channels_ker, num_channels_ker * 2, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_channels_ker * 2, num_channels_ker * 4, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_channels_ker * 4, num_channels_ker * 8, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True)
        )
        self.cls_conv = nn.Sequential(
            nn.Conv2d(num_channels_ker * 8, out_channels=32, kernel_size=4, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )

        self.src_conv = nn.Sequential(
            nn.Conv2d(num_channels_ker * 8, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=9, out_features=1)
        )

    def forward(self, x):
        x = self.main(x)
        out_src = self.src_conv(x)
        out_dist = self.cls_conv(x)
        return out_src, out_dist.view(out_dist.size(0), out_dist.size(1))


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

        print(' %dth network' % epoch)

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

        netGen = Generator(3 + 3, 3, self.num_channels_ker, self.norm, res_blocks_num=self.res_blocks_num, )
        netCritic = Critic(3, self.num_channels_ker, norm='inorm', num_classes=self.len_attrs, inp_size=self._out,
                           out_size=self._out, nrepeat=self.res_blocks_num)
        netGen = init_weights(netGen, init_type='normal', init_gain=0.02)
        netCritic = init_weights(netCritic, init_type='normal', init_gain=0.02)

        loss_GradPenalty = GradientPanaltyLoss().to(self.device)

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

                input = data[0]
                label_in = data[1]
                label_out = label_in[torch.randperm(label_in.size(0))]

                c_org = label_in.clone()
                c_trg = label_out.clone()

                x_real = input.to(self.device)
                c_trg = c_trg.to(self.device)
                c_org = c_org.to(self.device)
                label_org = label_in.to(self.device)
                label_trg = label_out.to(self.device)

                out_src, out_cls = netCritic(x_real)
                d_loss_real = -torch.mean(out_src)
                out_cls, label_org = self._tensor_to_float(out_cls, label_org)
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls[:, :label_org.size(1)], label_org,
                                                                size_average=False) / label_org.size(0)

                x_fake = netGen(x_real, c_trg)
                out_src, _ = netCritic(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = netCritic(x_hat)
                d_loss_gp = loss_GradPenalty(out_src, x_hat)

                d_loss = d_loss_real + d_loss_fake + self.wgt_cls * d_loss_cls + self.wgt_rec * d_loss_gp
                optimD.zero_grad()
                optimG.zero_grad()
                d_loss.backward()
                optimD.step()

                loss = dict()
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                if (i + 1) % self.n_critic_run == 0:
                    # backward netGen
                    x_fake = netGen(x_real, c_trg)
                    out_src, out_cls = netCritic(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    out_cls, label_trg = self._tensor_to_float(out_cls, label_trg)
                    g_loss_cls = F.binary_cross_entropy_with_logits(out_cls[:, :label_trg.size(1)], label_trg,
                                                                    size_average=False) / label_trg.size(0)

                    # Target-to-original domain.
                    x_reconst = netGen(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.wgt_rec * g_loss_rec + self.wgt_cls * g_loss_cls
                    optimD.zero_grad()
                    optimG.zero_grad()
                    g_loss.backward()
                    optimG.step()

                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                    wandb.log(loss)

                if show_results(self.num_freq_disp * 2):
                    input = transform_inv(input)
                    output = transform_inv(x_fake)
                    recon = transform_inv(x_reconst)

                    inp_attrs = set(
                        [self.index2attr[i] for i, att_val in enumerate(label_in[0].detach().view(-1)) if att_val == 1])
                    out_attrs = set([self.index2attr[i] for i, att_val in enumerate(label_out[0].detach().view(-1)) if
                                     att_val == 1])
                    ep = num_batch_train * (epoch - 1) + i
                    wandb.log({f'iter {ep}': [
                        wandb.Image(input[0], caption=f"input new attrs {' '.join(list(inp_attrs))}"),
                        wandb.Image(output[0],
                                    caption=f"output new attrs {' '.join(list(out_attrs))}"),
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
                    input = data[0]
                    label_in = data[1]
                    label_out = torch.zeros_like(label_in)
                    label_out[:, j] = 1

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
