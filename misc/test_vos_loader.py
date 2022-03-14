import config
import numpy as np
from dataset_factory.StylizedDAVIS import *
from dataset_factory.StylizedDAVISCinematograms import StylizedDAVISCinematograms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
import torch.utils.data as data
import cv2

def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std).permute(1,2,0)

def vis_pair(args, data1, data2, factor, sq_name='test', data_idx=0):
    batch_size = int(args.batch_size)
    for batch_idx in range(data1['Image'].shape[0]):
        if args.fwbwflow:
            for frame_idx in range(2):
                cv2.imwrite("tmp/%d/pair1_img_%s_%02d_%05d.png"%(factor, sq_name, frame_idx, data_idx*batch_size+batch_idx),
                        np.array(denorm(data1['Image'][batch_idx, frame_idx])*255)[:,:,::-1])
                cv2.imwrite("tmp/%d/pair1_flow_%s_%02d_%05d.png"%(factor, sq_name, frame_idx, data_idx*batch_size+batch_idx),
                        np.array(denorm(data1['Flow'][batch_idx, frame_idx, :3])*255)[:, :, ::-1])
                cv2.imwrite("tmp/%d/pair2_img_%s_%02d_%05d.png"%(factor, sq_name, frame_idx, data_idx*batch_size+batch_idx),
                        np.array(denorm(data2['Image'][batch_idx, frame_idx])*255)[:, :, ::-1])
                cv2.imwrite("tmp/%d/pair2_flow_%s_%02d_%05d.png"%(factor, sq_name, frame_idx, data_idx*batch_size+batch_idx),
                        np.array(denorm(data2['Flow'][batch_idx, frame_idx, :3])*255)[:, :, ::-1])

        else:
            cv2.imwrite("tmp/%d/pair1_img_%s_%05d.png"%(factor, sq_name, data_idx*batch_size+batch_idx),
                    np.array(denorm(data1['Image'][batch_idx])*255)[:,:,::-1])
            cv2.imwrite("tmp/%d/pair1_flow_%s_%05d.png"%(factor, sq_name, data_idx*batch_size+batch_idx),
                np.array(denorm(data1['Flow'][batch_idx])*255)[:, :, ::-1])
            cv2.imwrite("tmp/%d/pair2_img_%s_%05d.png"%(factor, sq_name, data_idx*batch_size+batch_idx),
                    np.array(denorm(data2['Image'][batch_idx])*255)[:, :, ::-1])
            cv2.imwrite("tmp/%d/pair2_flow_%s_%05d.png"%(factor, sq_name, data_idx*batch_size+batch_idx),
                    np.array(denorm(data2['Flow'][batch_idx])*255)[:, :, ::-1])
if __name__ == "__main__":

    args = config.load_args()
    args.fwbwflow = True

    #dataset = StylizedDAVISCinematograms(args, debug=True)
    dataset = StylizedDAVIS(args, debug=True)

    def collate_fn(batch):
        # Always return single first element
        return batch[0]
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    fg1 = plt.figure(1)
    fg2 = plt.figure(2)
    fg3 = plt.figure(3)
    fg4 = plt.figure(4)
    plt.ion()

    if not os.path.exists('tmp/'):
        os.makedirs('tmp/0')
        os.makedirs('tmp/1')
        os.makedirs('tmp/2')
        os.makedirs('tmp/3')

    batch_size = args.batch_size
    loader_itr = iter(loader)
    for it, (factor, sq1, sq2, sq_name, st1, st2) in enumerate(loader):
        print(it, ', Factor: ', factor)
        print(sq_name, ' ', st1, ' ', st2)
        sq_loader = data.DataLoader(PairedConcatDataset([sq1, sq2]),
                                batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers)
        for data_idx, data_pair in enumerate(sq_loader):
            data1 = data_pair[0]
            data2 = data_pair[1]

            vis_pair(args, data1, data2, factor, sq_name=sq_name, data_idx=data_idx)
