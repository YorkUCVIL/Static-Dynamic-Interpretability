import json
import pickle
from easydict import EasyDict as edict
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
import os
import random
import numpy as np
import torch.utils.data as data

from utils import *
from config import *
from dataset_factory.StylizedDAVIS import StylizedDAVIS, PairedConcatDataset
from dataset_factory.StylizedDAVISCinematograms import StylizedDAVISCinematograms
import models.vos_models as models
from misc.test_vos_loader import vis_pair

def get_dataset(opts):
    dst = None
    if opts.dataset == 'StylizedDAVIS':
        dst = StylizedDAVIS(opts)
    elif opts.dataset == 'StylizedDAVISCinematograms':
        dst = StylizedDAVISCinematograms(opts)
    return dst


def infer(model, loader, device, stages, variant, sampling_method, opts, ckpt_iteration,
          model_name):
    """Do Inference and estimate dim"""
    output_dict = {'example1': [], 'example2': []}
    factor_list = []


    model.eval()
    with torch.no_grad():
        for factor, sq1, sq2 in tqdm(loader):
            sq_loader = data.DataLoader(PairedConcatDataset([sq1, sq2]),
                                batch_size=int(opts.batch_size),
                                shuffle=False, num_workers=opts.num_workers)
            for i, data_pair in tqdm(enumerate(sq_loader)):
                #vis_pair(opts, data_pair[0], data_pair[1], factor)
                output_pair = []
                for data_ in data_pair:
                    if variant == "img_flow":
                        images = {}
                        images['Image'] = \
                            data_['Image'].to(device, dtype=torch.float32)
                        images['Flow'] = \
                            data_['Flow'].to(device, dtype=torch.float32)
                    elif variant == "img":
                        images = \
                            data_['Image'].to(device, dtype=torch.float32)
                    elif variant == "flow":
                        images = \
                            data_['Flow'].to(device, dtype=torch.float32)
                    else:
                        raise NotImplementedError

                    output = model(images, stages)[1]
                    output_pair.append(output)

                key = list(output_pair[0].keys())[0]
                current_shape = output_pair[0][key].shape[0]
                factor_list.append(np.array([factor]*current_shape))

                output_dict['example1'].append(output_pair[0])
                output_dict['example2'].append(output_pair[1])

#    with open('output_%s_%d.pkl'%(model_name, opts.random_seed), 'wb') as f:
#        pickle.dump(output_dict, f)
#    with open('factors_%s_%d.pkl'%(model_name, opts.random_seed), 'wb') as f:
#        pickle.dump(factor_list, f)
#    with open('output_%s_%d.pkl'%(model_name, opts.random_seed), 'rb') as f:
#        output_dict = pickle.load(f)
#    with open('factors_%s_%d.pkl'%(model_name, opts.random_seed), 'rb') as f:
#        factor_list = pickle.load(f)

    if not os.path.exists('dim_outputs/vos_models/final/%s'%model_name):
        os.makedirs('dim_outputs/vos_models/final/%s'%model_name)
    if not os.path.exists('dim_outputs/vos_models/idv_scores/%s'%model_name):
        os.makedirs('dim_outputs/vos_models/idv_scores/%s'%model_name)
    if not os.path.exists('dim_outputs/vos_models/joint_encoding/%s'%model_name):
        os.makedirs('dim_outputs/vos_models/joint_encoding/%s'%model_name)
    if not os.path.exists('dim_outputs/vos_models/stats_tables_newformat'):
        os.makedirs('dim_outputs/vos_models/stats_tables_newformat')

    for stage in stages:
        current_output_dict = {'example1':[], 'example2':[]}
        for key, value in output_dict.items():
            for element in value:
                for key_stage, feats in element.items():
                    if key_stage == stage:
                        current_output_dict[key].append(feats)

        if len(current_output_dict['example1']) == 0:
            # Not found stages
            continue

        opts.stg = stage
        opts.model = model_name
        dims, dims_percent, idv_scores = dim_est(current_output_dict, factor_list, opts, return_idv_scores=True)
        with open('dim_outputs/vos_models/idv_scores/%s/%s.pkl'%(model_name, stage), 'wb') as f:
            pickle.dump(idv_scores, f)

        with open('dim_outputs/vos_models/final/%s/%s_%s_%03d_%s.txt'%(\
                   model_name, model_name, ckpt_iteration, opts.random_seed, opts.pretrain_type), 'a') as f:
            f.write('Stage: ' + stage + ' , Dimensions: ' + \
                    str(dims) + ' ' + str(dims_percent) + '\n')
            print('Stage: ' + stage + ' , Dimensions: ' + \
                    str(dims) + ' ' + str(dims_percent) + '\n')


def load_and_infer_checkpoint(generic_opts, opts, model, ckpt_file, infer_loader, device):

    if generic_opts.checkpoint != "imgnet":
        if opts.model == "matnet":
            model.load_ckpt(ckpt_file)
            print('Loaded MATNet Checkpoint from ', ckpt_file)
            model = nn.DataParallel(model)
        elif opts.model == "rtnet":
            model = nn.DataParallel(model)
            checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["state_dict"])
            print('Loaded RTNet Checkpoint from ', ckpt_file)
        elif opts.model == "twostream_deeplabv3plus_resnet101":
            checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            print('Loaded Checkpoint from %s'%ckpt_file)


    ckpt_iteration = 'final'

    model.to(device)

    stage_map = {0: 'conv1,app_stream', 1: 'layer1,app_stream', 2: 'layer2,app_stream',
                 3: 'layer3,app_stream', 4: 'layer4,app_stream', 5: 'conv1,mot_stream',
                 6: 'layer1,mot_stream', 7: 'layer2,mot_stream', 8:'layer3,mot_stream',
                 9: 'layer4,mot_stream', 10: 'layer1,sensor_fusion', 11: 'layer2,sensor_fusion',
                 12: 'layer3,sensor_fusion', 13: 'layer4,sensor_fusion', 14: 'conv1,sensor_fusion',
                 15: 'layerf,sensor_fusion'}

    if opts.model == "sam":
        # Include conv1 sensor fusion
        stages = list(range(15))
    else:
        stages = list(range(14))

    #stages = [10, 11, 12, 13, 15]
    model_name = opts.model
    if model_name == 'matnet':
        matnet_args = ''
        for k, v in opts.extra_args.items():
            matnet_args += k + '_' + v + '_'
        model_name += '_' + matnet_args[:-1]
    elif model_name == 'twostream_deeplabv3plus_resnet101':
        model_name += '_' + generic_opts.trained_on

    stage_names = [stage_map[stage] for stage in stages]
    infer(model, infer_loader, device, stage_names, opts.variant, opts.sampling_method,
          generic_opts, ckpt_iteration, model_name)

def main():
    generic_opts = load_args()

    # Load Config File
    with open(generic_opts.cfg_file, 'r')as f:
        json_dic = json.load(f)
    opts = edict(json_dic)

    # Set Device
    if generic_opts.device != -1:
        opts.gpu_id = str(generic_opts.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    opts.random_seed = generic_opts.random_seed
    torch.manual_seed(generic_opts.random_seed)
    np.random.seed(generic_opts.random_seed)
    random.seed(generic_opts.random_seed)

    # Setup dataloader
    if hasattr(opts, 'fwbwflow'):
        generic_opts.fwbwflow = opts.fwbwflow
    infer_dst = get_dataset(generic_opts)
    def collate_fn(batch):
        # Always return single first element
        return batch[0]
    infer_loader = data.DataLoader(
        infer_dst, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn)
    print("Dataset: %s, Infer set: %d" %
          (generic_opts.dataset, len(infer_dst)))

    # Set up model
    if torch.__version__ == "0.4.0":
        model_map = {'sam': models.sam}
    else:
        model_map = {
            'twostream_deeplabv3plus_resnet101': models.twostream_deeplabv3plus_resnet101,
            'matnet': models.matnet,
            'rtnet': models.rtnet,
            'rtnet34': models.rtnet34
        }

    if not hasattr(opts, 'extra_args'):
        opts.extra_args = {}

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,
                                  fuse_early=opts.fuse_early, pretrain_motionstream=opts.pretrain_motionstream,
                                  fuse_bnorm=opts.fuse_bnorm, extra_args=opts.extra_args)

    load_and_infer_checkpoint(generic_opts, opts, model, generic_opts.checkpoint,
                              infer_loader, device)

if __name__ == '__main__':
    main()
