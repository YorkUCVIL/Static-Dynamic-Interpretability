#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

import models.ar_models.SlowFast.slowfast.utils.checkpoint as cu
from models.ar_models.SlowFast.slowfast.config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )

    parser.add_argument('--save_dir', default='dim_outputs', help="dataset to use for dim estimation")
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--stylized_data_dir',
                        default='/local/data1/mkowal/data/stylized_datasets/val_stylized_Diving48',
                        help="Path to stylized dataset directory and json files")
    parser.add_argument('--cfg_file', default='configs/twostreamv3plus_davis.json')

    # data
    parser.add_argument('--dataset', default='Diving48',
                        help="dataset to use for dim estimation (StylizedActivityNet | ssv2 | Diving48)")
    parser.add_argument('--app_shuffle', default=True, help="shuffle the appearance pair frames")
    parser.add_argument('--use_normal_app', default=False, help="use normal videos for the appearance pair frames")
    parser.add_argument('--m_same', default=True, help="Use same video for motion pair")

    # model
    parser.add_argument('--model', default='i3d', help="model to do dimension estimation on")
    parser.add_argument('--stg', default=None, help="stage of network to analyze")
    parser.add_argument('--path', default='fast', help="Slowfast path to use (cat | slow | fast)")
    parser.add_argument('--fuse', default=True, help="Slowfast to fuse before returning midlayer")
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--trained_on', default='davis', type=str)

    # custom evaluation parameters
    parser.add_argument('--checkpoint', default='', help="path to checkpoint")
    parser.add_argument('--stage', default=5, type=int, help="stage or layer to perform estimation on")

    # dim estimation
    parser.add_argument('--n_factors', default=3, help="number of factors (including residual)")
    parser.add_argument('--styles', default='1,2,3,4', help="ids of styles")
    parser.add_argument('--residual_index', default=2, help="index of residual factor (usually last)")
    parser.add_argument('--joint_encoding', default=False, type=bool,
                        help="Present figure of motion-appearance joint encoding neurons")
    parser.add_argument('--joint_encoding_thresh', default=0.5,
                        help="Thrershold of motion-appearance joint encoding neurons")

    # data loading details
    parser.add_argument('--batch_size', default=2, help="batch size during evaluation")
    parser.add_argument('--image_size', default=256, help="image size during evaluation")
    parser.add_argument('--n_sample_frames', default=64, help="number of frames to sample from video during training")
    parser.add_argument('--n_examples', default=3000,
                        help="number of examples to use for estimation, should lower this if you run out of memory")

    # validation params
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')

    # computing
    parser.add_argument('--device', default=0, type=int, help="gpu id")
    parser.add_argument('--num_workers', default=0, type=int, help="number of CPU threads")

    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg
