from os.path import dirname, join, basename, isfile
import os, random, cv2, argparse
from glob import glob
import subprocess

from tqdm import tqdm
import audio
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import wandb

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual
from hparams import hparams, get_image_list
from color_syncnet_train import _load

wandb.init(project='Wav2Lip-project', reinit=True)
STEP = 0
EPOCH = 0
T = 5
MEL_STEP_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', type=str)
args = parser.parse_args()


class Dataset(object):
    def __init__(self, split):
        self.videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def __len__(self):
        return len(self.videos)

    def read_window(self, window_fnames):
        if window_fnames
            window = []
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    return None
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as exception:
                    return None
                window.append(img)
            return window
        return None

    def get_window(self, start):
        start_id = self.get_frame_id(start)
        vidname = dirname(start)

        window_fnames = []
        for frame_id in range(start_id, start_id + T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop(self, spec, start_frame):
        start_frame_num = start_frame
        if not type(start_frame) == int: start_frame_num = self.get_frame_id(start_frame)
        start_id = int(80. * (start_frame_num / float(hparams.fps)))
        return spec[start_id: start_id + MEL_STEP_SIZE, :]

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.videos) - 1)
            names = self.videos[idx]
            img_names = list(glob(join(names, '*.jpg')))
            if len(img_names) <= 3 * T: continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None: continue

            window = self.read_window(window_fnames)
            wrong_window = self.read_window(wrong_window_fnames)

            if not wrong_window or not window: continue
            try:
                wavpath = join('vidname', "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as exception:
                continue

            mel = self.crop(orig_mel.copy(), img_name)

            if (mel.shape[0] != MEL_STEP_SIZE):
                continue

            indiv_mels = self.segment(orig_mel.copy(), img_name)
            indiv_mels = []
            start_n = self.get_frame_id(img_name) + 1
            if start_n - 2 < 0: return None
            for i in range(start_n, start_n + T):
                m = self.crop(orig_mel.copy(), i - 2)
                if m.shape[0] != MEL_STEP_SIZE:
                    return None
                indiv_mels.append(m.T)

            if indiv_mels is None: continue
            indiv_mels = np.asarray(indiv_mels)

            window = np.asarray(window) / 255.
            window = np.transpose(window, (3, 0, 1, 2))
            y = torch.FloatTensor(window.copy())
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = np.asarray(wrong_window) / 255.
            wrong_window = np.transpose(wrong_window, (3, 0, 1, 2))
            x = np.concatenate([window, wrong_window], axis=0)

            return torch.FloatTensor(x), torch.FloatTensor(indiv_mels).unsqueeze(1), torch.FloatTensor(mel.T).unsqueeze(
                0), y


def sample(x, g, gt, STEP, checkpoint_dir):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(STEP))
    if not os.path.exists(folder): os.mkdir(folder)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    concatenated = np.concatenate((x[..., 3:], x[..., :3], g, gt), axis=-2)
    for idx, batch in enumerate(concatenated):
        for t in range(len(batch)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, idx, t), batch[t])


def cos_loss(a, v, y):
    return nn.BCELoss()(nn.functional.cosine_similarity(a, v).unsqueeze(1), y)


def sync_loss(mel, g):
    g = g[:, :, :, g.size(3) // 2:]
    a, v = syncnet(mel, torch.cat([g[:, :, i] for i in range(T)], dim=1))
    return cos_loss(a, v, torch.ones(g.size(0), 1).float().to(device))


def train(device, gen, disc, dataloader_train, dataloader_test, gen_optimizer, disc_optimizer, checkpoint_dir=None):
    global STEP, EPOCH
    resumed_step = STEP

    while EPOCH < hparams.nepochs:
        wandb.log({'G_train/epoch': EPOCH})
        running_sync_loss = 0.
        running_l1_loss = 0.
        disc_loss = 0.
        running_perceptual_loss = 0.
        running_disc_real_loss = 0.
        running_disc_fake_loss = 0.

        for step, (x, indiv_mels, mel, gt) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            gen.train()
            disc.train()

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            generated = gen(indiv_mels, x)
            l1loss = nn.L1Loss()(generated, gt)
            prec_loss = disc.perceptual_forward(generated)
            sync = sync_loss(mel, generated)
            loss = hparams.syncnet_wt * sync + hparams.disc_wt * prec_loss + (
                        1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss
            loss.backward()
            gen_optimizer.step()

            disc_optimizer.zero_grad()
            pred = disc(generated)
            disc_fake = F.binary_cross_entropy(disc(generated.detach()), torch.zeros((len(pred), 1)).to(device))
            disc_real = F.binary_cross_entropy(disc(gt), torch.ones((len(pred), 1)).to(device))
            disc_loss = disc_fake + disc_real
            disc_loss.backward()
            disc_optimizer.step()

            running_disc_real_loss += disc_real.item()
            running_disc_fake_loss += disc_fake.item()

            if STEP % hparams.checkpoint_interval == 0:
                sample(x, generated, gt, STEP, checkpoint_dir)
            STEP += 1
            cur_session_steps = STEP - resumed_step

            running_l1_loss += l1loss.item()

            if hparams.disc_wt > 0:
                running_perceptual_loss += prec_loss.item()
            if hparams.syncnet_wt > 0:
                running_sync_loss += sync.item()

            if STEP == 1 or STEP % hparams.checkpoint_interval == 0:
                save_checkpoint(gen, gen_optimizer, STEP, checkpoint_dir, EPOCH)
                save_checkpoint(disc, disc_optimizer, STEP, checkpoint_dir, EPOCH, prefix='disc_')

            if STEP % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(dataloader_test, STEP, device, gen, disc)
                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.03)

            wandb.log({
                'G_train/L1': running_l1_loss / (step + 1),
                'G_train/Sync': running_sync_loss / (step + 1),
                'G_train/Percep': running_perceptual_loss / (step + 1),
                'G_train/Fake': running_disc_fake_loss / (step + 1),
                'G_train/Real': running_disc_real_loss / (step + 1)
            })

        EPOCH += 1


def eval_model(dataloader_test, STEP, device, gen, disc):
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    for step, (x, indiv_mels, mel, gt) in enumerate((dataloader_test)):
        gen.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        generated = gen(indiv_mels, x)
        pred = disc(generated)
        disc_fake = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

        pred = disc(gt)
        disc_real = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

        running_disc_real_loss.append(disc_real.item())
        running_disc_fake_loss.append(disc_fake.item())

        _sync_loss = sync_loss(mel, generated)

        perceptual_loss = 0.

        if hparams.disc_wt > 0.:
            perceptual_loss = disc.perceptual_forward(generated)

        l1loss = nn.L1Loss()(generated, gt)
        loss = hparams.syncnet_wt * _sync_loss + hparams.disc_wt * perceptual_loss + (
                    1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

        running_l1_loss.append(l1loss.item())
        running_sync_loss.append(_sync_loss.item())

        running_perceptual_loss.append(perceptual_loss.item())

        wandb.log({
            'G_eval/L1': sum(running_l1_loss) / len(running_l1_loss),
            'G_eval/Sync': sum(running_sync_loss) / len(running_sync_loss),
            'G_eval/Percep': sum(running_perceptual_loss) / len(running_perceptual_loss),
            'G_eval/Fake': sum(running_disc_fake_loss) / len(running_disc_fake_loss),
            'G_eval/Real': sum(running_disc_real_loss) / len(running_disc_real_loss)
        })

        return sum(running_sync_loss) / len(running_sync_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, STEP))
    optimizer_state = None
    if hparams.save_optimizer_state:
        optimizer_state = optimizer.state_dict()

    subprocess.call(f'rm -rf {checkpoint_dir}/{prefix}*checkpoint_step*.pth', shell=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "STEP": step,
        "EPOCH": epoch,
    }, checkpoint_path)


def load_checkpoint(path, model, optimizer):
    global STEP
    global EPOCH
    new_s = {}
    for key, val in _load(path)["state_dict"].items(): new_s[key.replace('module.', '')] = val
    model.load_state_dict(new_s)
    return model


if __name__ == "__main__":
    dataloader_train = data_utils.DataLoader(Dataset('train'), batch_size=hparams.batch_size, shuffle=True,
                                             num_workers=hparams.num_workers)
    dataloader_test = data_utils.DataLoader(Dataset('val'), batch_size=hparams.batch_size, num_workers=4)

    gen = Wav2Lip()
    disc = Wav2Lip_disc_qual()
    gen.to(device)
    disc.to(device)

    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, hparams.gpu_ids)
        disc = nn.DataParallel(disc, hparams.gpu_ids)
        cudnn.benchmark = hparams.benchmark

    gen_optimizer = optim.Adam(gen.parameters(), lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc.parameters(), lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, gen, gen_optimizer)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer)

    load_checkpoint('syncnet_checkpoints/checkpoint_step000018100.pth', syncnet, None)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    train(device, gen, disc, dataloader_train, dataloader_test, gen_optimizer, disc_optimizer,
          checkpoint_dir='checkpoint_dir')
