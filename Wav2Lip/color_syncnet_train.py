from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import subprocess
import os, random, cv2, argparse
from hparams import hparams, get_image_list

import wandb

wandb.init(project='Wav2Lip-project', reinit=True)

parser = argparse.ArgumentParser()

parser.add_argument("--root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()

STEP = 0
EPOCH = 0
T = 5
MEL_STEP_SIZE = 16


class videosDataset(object):
    def __init__(self, split):
        self.videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        while True:
            idx = random.randint(0, len(self.videos) - 1)
            vidname = self.videos[idx]
            
            read_flg = True

            names = list(glob(join(vidname, '*.jpg')))
            if len(names) <= 3 * T:
                continue

            img_name = random.choice(names)
            wrong_img_name = random.choice(names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                picked_frame = img_name
            else:
                y = torch.zeros(1).float()
                picked_frame = wrong_img_name


            start_id = self.get_frame_id(picked_frame)
            curname = dirname(picked_frame)

            window_fnames = []
            for frame_id in range(start_id, start_id + T):
                frame = join(curname, '{}.jpg'.format(frame_id))
                if not isfile(frame):
                    return None
                window_fnames.append(frame)

            if window_fnames is None:
                continue

            frames = []
            
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    read_flg = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    read_flg = False
                    break

                frames.append(img)

            if not read_flg: continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as exception:
                continue


            start = self.get_frame_id(img_name)
            start = int(80. * (start / float(hparams.fps)))

            mel = orig_mel[start: start + MEL_STEP_SIZE, :]

            if (mel.shape[0] != MEL_STEP_SIZE):
                continue

            x = (np.concatenate(window, axis=2) / 255.).transpose(2, 0, 1)
            return torch.FloatTensor(x[:, x.shape[1] // 2:]), torch.FloatTensor(mel.T).unsqueeze(0), y


def cos_loss(a, v, y):
    return nn.BCELoss()(nn.functional.cosine_similarity(a, v).unsqueeze(1), y)


def train(device, model, dataloader_train, dataloader_test, optimizer, checkpoint_dir=None):
    global STEP, EPOCH
    resumed_step = STEP

    while EPOCH < hparams.nepochs:
        running_loss = 0.
        for step, (x, mel, y) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            model.train()
            optimizer.zero_grad()

            x = x.to(device)
            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cos_loss(a, v, y)
            loss.backward()
            optimizer.step()

            STEP += 1
            cur_session_steps = STEP - resumed_step
            running_loss += loss.item()

            if STEP == 1 or STEP % hparams.syncnet_checkpoint_interval == 0:
                save_checkpoint(model, optimizer, STEP, checkpoint_dir, EPOCH)

            if STEP % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_(dataloader_test, STEP, device, model)
            if step % hparams.log_interval == 0:
                wandb.log({'Train cosine loss': running_loss / (step + 1)})

        EPOCH += 1


def eval_(dataloader_test, STEP, device, model):
    losses = []
    for data in dataloader_test:
        model.eval()
        x, mel, y = data
        x = x.to(device)
        y = y.to(device)
        mel = mel.to(device)
        
        a, v = model(mel, x)
        loss = cos_loss(a, v, y)
        losses.append(loss.item())

    wandb.log({'Eval cosine loss': sum(losses) / len(losses)})


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(STEP))
    
    optimizer_state = None
    if hparams.save_optimizer_state:
        optimizer_state = optimizer.state_dict()

    subprocess.call(f'rm -rf {checkpoint_dir}/*', shell=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "STEP": step,
        "EPOCH": epoch,
    }, checkpoint_path)

def load_checkpoint(path, model, optimizer):
    global STEP
    global EPOCH
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint["optimizer"]
    if optimizer_state is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    STEP = checkpoint["STEP"]
    EPOCH = checkpoint["EPOCH"]

    return model, optimizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): 
        os.mkdir(checkpoint_dir)

    dataloader_train = data_utils.DataLoader(videosDataset('train'), batch_size=hparams.syncnet_batch_size, shuffle=True, num_workers=hparams.num_workers)
    dataloader_test = data_utils.DataLoader(videosDataset('val'), batch_size=hparams.syncnet_batch_size, num_workers=8)

    model = SyncNet()
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, hparams.gpu_ids)
        cudnn.benchmark = hparams.benchmark

    optimizer = optim.Adam(model.parameters() ,hparams.syncnet_lr)
    if checkpoint_path:
        model, optimizer = load_checkpoint(checkpoint_path, model, optimizer)

    train(device, model, dataloader_train, dataloader_test, optimizer, checkpoint_dir=checkpoint_dir)
