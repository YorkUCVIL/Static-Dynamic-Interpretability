import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Dimension estimation')

    # saving, debuging, and configs
    parser.add_argument('--save_dir', default='dim_outputs', help="dataset to use for dim estimation")
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--stylized_data_dir',
                        default='/media/ssd1/m3kowal/Stylized_ActivityNet',
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

    args = parser.parse_args()
    return args
