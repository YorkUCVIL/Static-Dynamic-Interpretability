import config
import numpy as np
from dataset_factory.StylizedDAVIS import *
from dataset_factory.StylizedDAVISCinematograms import StylizedDAVISCinematograms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
import torch.utils.data as data
from utils import get_dataloader, get_model
from tqdm import tqdm
import cv2
from torchvision.transforms.functional import normalize

def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std).permute(1,2,0) * 255

def save_sequence(slow_seq, fast_seq, save_dir, bidx, prefix):
    for fidx in range(slow_seq.shape[1]):
        slowdir = os.path.join(save_dir, "%s_slow_%02d_%02d.png"%(prefix, bidx, fidx))
        cv2.imwrite(slowdir, np.array(denorm(slow_seq[:, fidx])))

    if fast_seq is not None:
        for fidx in range(fast_seq.shape[1]):
            fastdir = os.path.join(save_dir, "%s_fast_%02d_%02d.png"%(prefix, bidx, fidx))
            cv2.imwrite(fastdir, np.array(denorm(fast_seq[:, fidx])))

if __name__ == "__main__":

    args = config.load_args()

    _ = get_model(args)
    dataloader = get_dataloader(args)

    if not os.path.exists('tmp/'):
        os.makedirs('tmp/0')
        os.makedirs('tmp/1')
        os.makedirs('tmp/2')
        os.makedirs('tmp/3')

    for i, (factor, example1, example2) in enumerate(tqdm(dataloader)):
        print(i, ', Factor: ', factor)

        data1 = example1
        data2 = example2
        for batch_idx in range(data1[0].shape[0]):
            if factor[batch_idx] != 2:
                continue

            if 'slowfast' in args.model:
                save_sequence(data1[0][batch_idx], data1[1][batch_idx], "tmp/%d/"%factor[batch_idx],
                              i*2 + batch_idx, "data1")
                save_sequence(data2[0][batch_idx], data2[1][batch_idx], "tmp/%d/"%factor[batch_idx],
                              i*2 + batch_idx, "data2")
            else:
                save_sequence(data1[0][batch_idx], None, "tmp/%d/"%factor[batch_idx], batch_idx, "data1")
                save_sequence(data2[0][batch_idx], None, "tmp/%d/"%factor[batch_idx], batch_idx, "data2")

