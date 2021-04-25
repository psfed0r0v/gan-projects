import sys
from os import listdir, path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp
import face_detection

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=4, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=128, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", default='data', type=str)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default='processed_data', type=str)
args = parser.parse_args()

face_alignment = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id)) for id in range(args.ngpu)]




def prepare_audio(vfile, args):
    audir = path.join(args.preprocessed_root, vfile.split('/')[-2], os.path.basename(vfile).split('.')[0])
    os.makedirs(audir, exist_ok=True)
    subprocess.call('ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(vfile, path.join(audir, 'audio.wav')), shell=True)

def prepare_video(vfile, args, gpu_id):
    video_stream = cv2.VideoCapture(vfile)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    imdir = path.join(args.preprocessed_root, vfile.split('/')[-2], os.path.basename(vfile).split('.')[0])
    os.makedirs(imdir, exist_ok=True)
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
    for batch in batches:
        preds = face_alignment[gpu_id].get_detections_for_batch(np.asarray(batch))
        for j, f in enumerate(preds):
            if not f: continue
            x_1, y_1, x_2, y_2 = f
            img = batch[j][y_1:y_2, x_1:x_2]
            cv2.imwrite(path.join(imdir, '{}.jpg'.format(j)), img)


def handler(job):
    vfile, args, gpu_id = job
    prepare_video(vfile, args, gpu_id)

if __name__ == '__main__':
    filelist = glob(path.join(args.data_root, '*/*/*.mp4'))
    filelist = filelist[:int(len(filelist) * 0.1)]
    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


    for vfile in tqdm(filelist):
        prepare_audio(vfile, args)
