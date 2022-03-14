import torch
import json
import pickle
from easydict import EasyDict as edict
import torch.nn as nn
import random
import numpy as np
from fvcore.nn.flop_count import flop_count
import os

from config import *
import models.vos_models as models

def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count

def infer(model, device, stages, variant, sampling_method, opts, ckpt_iteration,
          model_name):
    if 'rtnet' in model_name:
        inputs = ({'Flow': torch.zeros(2, 2, 6, 240, 427).cuda(), 'Image': torch.zeros(2, 2, 3, 240, 427).cuda()}, [])
    else:
        inputs = ({'Flow': torch.zeros(2, 3, 240, 427).cuda(), 'Image': torch.zeros(2, 3, 240, 427).cuda()}, [])
    gflops, _ = flop_count(model, inputs)
    total_flops = 0
    for k, v in gflops.items():
        total_flops += v
    nparams = params_count(model)
    print('GFlop Count is ', total_flops, 'G and Number of Parameters is ', nparams/1e6, 'M')

def load_and_infer_checkpoint(generic_opts, opts, model, ckpt_file, device):

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

    if generic_opts.iterate_checkpoint:
        ckpt_iteration = ckpt_file.split('/')[-1].split('_')[-1].split('.')[0]
    else:
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

    stages = [10, 11, 12, 13, 15]
    model_name = opts.model
    if model_name == 'matnet':
        matnet_args = ''
        for k, v in opts.extra_args.items():
            matnet_args += k + '_' + v + '_'
        model_name += '_' + matnet_args[:-1]

    stage_names = [stage_map[stage] for stage in stages]
    infer(model, device, stage_names, opts.variant, opts.sampling_method,
          generic_opts, ckpt_iteration, model_name)

def main():
    generic_opts = load_args()

    # Load Config File
    with open(generic_opts.cfg_file, 'r')as f:
        json_dic = json.load(f)
    opts = edict(json_dic)

    # Set Device
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

    load_and_infer_checkpoint(generic_opts, opts, model, generic_opts.checkpoint, device)

if __name__ == '__main__':
    main()
