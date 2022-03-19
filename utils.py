import torch
if torch.__version__ != "0.4.0":
    from models.ar_models.SlowFast.slowfast.models.build import *
    from models.ar_models.SlowFast.slowfast.utils.parser import load_config, parse_args
    from models.ar_models.SlowFast.slowfast.config.defaults import assert_and_infer_cfg
    import models.ar_models.SlowFast.slowfast.utils.checkpoint as cu

    # MViT repo
    from models.ar_models.MVIT_SlowFast.slowfast.models.build import build_model as mvit_build_model
    from models.ar_models.MVIT_SlowFast.slowfast.utils.parser import load_config as mvit_load_config
    from models.ar_models.MVIT_SlowFast.slowfast.utils.parser import parse_args as mvit_parse_args
    from models.ar_models.MVIT_SlowFast.slowfast.config.defaults import assert_and_infer_cfg as mvit_assert_and_infer_cfg
    import models.ar_models.MVIT_SlowFast.slowfast.utils.checkpoint as mvit_cu

    from dataset_factory.StylizedActivityNet import *
    from dataset_factory.ssv2 import *
    from dataset_factory.Diving48 import *

    from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
    from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
    from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle
import csv

def get_model(args):
    if args.dataset == 'Diving48':
        num_classes = 48
    elif args.dataset == 'ssv2':
        num_classes = 174
    elif args.dataset == 'StylizedActivityNet':
        num_classes = 400 # kinetics trained models
    if args.model == 'c2d':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/C2D_8x8_R50_IN1K.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'i3d':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/I3D_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'i3d_nln':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/I3D_NLN_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'slow_r50_8x8':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/SLOW_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'slow_ssv2_r50_8x8':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/SSv2/pytorchvideo/SLOW_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'slowfast_r50_8x8':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        # fast params
        args.fast_sampling_rate = cfg.DATA.SAMPLING_RATE
        args.fast_num_frames = cfg.DATA.NUM_FRAMES
        args.alpha = cfg.SLOWFAST.ALPHA
        # slow params
        args.slow_sampling_rate = cfg.DATA.SAMPLING_RATE * args.alpha
        args.slow_num_frames = int(cfg.DATA.NUM_FRAMES / args.alpha)
    elif args.model == 'slowfast_ssv2_r50_8x8':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/SSv2/pytorchvideo/SLOWFAST_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        # fast params
        args.fast_sampling_rate = cfg.DATA.SAMPLING_RATE
        args.fast_num_frames = cfg.DATA.NUM_FRAMES
        args.alpha = cfg.SLOWFAST.ALPHA
        # slow params
        args.slow_sampling_rate = cfg.DATA.SAMPLING_RATE * args.alpha
        args.slow_num_frames = int(cfg.DATA.NUM_FRAMES / args.alpha)
    elif args.model == 'slow_r50_4x16':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/SLOW_4x16_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    elif args.model == 'slowfast_r50_4x16':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/c2/SLOWFAST_4x16_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        # fast params
        args.fast_sampling_rate = cfg.DATA.SAMPLING_RATE
        args.fast_num_frames = cfg.DATA.NUM_FRAMES
        args.alpha = cfg.SLOWFAST.ALPHA
        # slow params
        args.slow_sampling_rate = cfg.DATA.SAMPLING_RATE * args.alpha
        args.slow_num_frames = int(cfg.DATA.NUM_FRAMES / args.alpha)
    elif args.model == 'slowfast_ssv2_16x8':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/SSv2/SLOWFAST_16x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        # fast params
        args.fast_sampling_rate = cfg.DATA.SAMPLING_RATE
        args.fast_num_frames = cfg.DATA.NUM_FRAMES
        args.alpha = cfg.SLOWFAST.ALPHA
        # slow params
        args.slow_sampling_rate = cfg.DATA.SAMPLING_RATE * args.alpha
        args.slow_num_frames = int(cfg.DATA.NUM_FRAMES / args.alpha)
    elif args.model == 'slowfast_ssv2_16x8_multigrid':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/SSv2/SLOWFAST_16x8_R50_multigrid.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        # fast params
        args.fast_sampling_rate = cfg.DATA.SAMPLING_RATE
        args.fast_num_frames = cfg.DATA.NUM_FRAMES
        args.alpha = cfg.SLOWFAST.ALPHA
        # slow params
        args.slow_sampling_rate = cfg.DATA.SAMPLING_RATE * args.alpha
        args.slow_num_frames = int(cfg.DATA.NUM_FRAMES / args.alpha)
    elif args.model == 'x3d_xs':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/X3D_XS.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
        args.n_sample_frames = args.sampling_rate * args.num_frames
        args.image_size = cfg.DATA.TEST_CROP_SIZE
        args.scale_size = cfg.DATA.TRAIN_JITTER_SCALES[0]
    elif args.model == 'x3d_s':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/X3D_S.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
        args.n_sample_frames = args.sampling_rate * args.num_frames
        args.image_size = cfg.DATA.TEST_CROP_SIZE
        args.scale_size = cfg.DATA.TRAIN_JITTER_SCALES[0]
    elif args.model == 'x3d_m':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/X3D_M.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
        args.n_sample_frames = args.sampling_rate * args.num_frames
        args.image_size = cfg.DATA.TEST_CROP_SIZE
        args.scale_size = cfg.DATA.TRAIN_JITTER_SCALES[0]
    elif args.model == 'x3d_l':
        model_args = parse_args()
        model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/X3D_L.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
        args.n_sample_frames = args.sampling_rate * args.num_frames
        args.image_size = cfg.DATA.TEST_CROP_SIZE
        args.scale_size = cfg.DATA.TRAIN_JITTER_SCALES[0]
    elif args.model == 'mvit':
        model_args = mvit_parse_args()
        model_args.cfg_file = 'models/ar_models/MVIT_SlowFast/configs/Kinetics/MVIT_B_16x4_CONV.yaml'
        cfg = mvit_load_config(model_args)
        cfg = mvit_assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = mvit_build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        mvit_cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
        args.n_sample_frames = args.sampling_rate*args.num_frames
        args.image_size = 224
    elif args.model == 'timesformer':
        from models.ar_models.TimeSformer.timesformer.models.vit import TimeSformer
        if args.dataset == 'ssv2':
            model_path = 'ar_models/TimeSformer/timesformer/models/ar_models/checkpoints/TimeSformer_divST_8_224_SSv2.pyth'
            print('Loading weights from: {}'.format(model_path))
            model = TimeSformer(img_size=224, num_classes=174, num_frames=16, attention_type='divided_space_time',
                                pretrained_model=model_path)
            args.sampling_rate = 8
            args.n_sample_frames = 8 * 16
            args.num_frames = 16
            args.image_size = 224
        else:
            model_path = 'models/ar_models/TimeSformer/timesformer/models/checkpoints/TimeSformer_divST_8x32_224_K400.pyth'
            print('Loading weights from: {}'.format(model_path))
            model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                                pretrained_model=model_path)
            args.sampling_rate = 32
            args.n_sample_frames = 8*32
            args.num_frames = 8
            args.image_size = 224
    elif args.model == 'fast':
        model_args = parse_args()
        if args.dataset == 'Diving48':
            model_args.cfg_file = 'models/ar_models/SlowFast/configs/Diving48/FAST_8x8_R50.yaml'
        elif args.dataset == 'ssv2':
            model_args.cfg_file = 'models/ar_models/SlowFast/configs/SSv2/FAST_8x8_R50.yaml'
        elif args.dataset == 'StylizedActivityNet':
            model_args.cfg_file = 'models/ar_models/SlowFast/configs/Kinetics/FAST_8x8_R50.yaml'
        cfg = load_config(model_args)
        cfg = assert_and_infer_cfg(cfg)
        cfg.MODEL.NUM_CLASSES = num_classes
        model = build_model(cfg)
        if len(args.checkpoint) > 0:
            cfg.TRAIN.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
            cfg.TEST.CHECKPOINT_TYPE = 'pytorch'
            cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
        cu.load_test_checkpoint(cfg, model)
        args.sampling_rate = cfg.DATA.SAMPLING_RATE
        args.num_frames = cfg.DATA.NUM_FRAMES
    return model

def get_dataloader(args):
    if args.dataset == 'StylizedActivityNet':
        dataset = StylizedActivityNet(args)
    if args.dataset == 'ssv2':
        dataset = SSV2(args)
    if args.dataset == 'Diving48':
        dataset = Diving48(args)

    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return dataloader


class Distribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean = torch.chunk(parameters, 1, dim=1)
        self.deterministic = deterministic

    def sample(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = self.mean + self.std*torch.randn(self.mean.shape).to(device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5*torch.sum(torch.pow(self.mean, 2)
                        + self.var - 1.0 - self.logvar,
                        dim=[1,2,3])
            else:
                return 0.5*torch.sum(
                        torch.pow(self.mean - other.mean, 2) / other.var
                        + self.var / other.var - 1.0 - self.logvar + other.logvar,
                        dim=[1,2,3])

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0*np.pi)
        return 0.5*torch.sum(
                logtwopi+self.logvar+torch.pow(sample-self.mean, 2) / self.var,
                dim=[1,2,3])

    def mode(self):
        return self.mean

def softmax_fn(scores, use_max=False):
    m = np.max(scores)
    if use_max:
        e = np.exp(scores-m)
    else:
        e = scores
    softmaxed = e / np.sum(e)
    return softmaxed

def dim_est(output_dict, factor_list, args, return_idv_scores=False):
    # grab flattened factors, examples
    za = np.concatenate(output_dict['example1'])
    zb = np.concatenate(output_dict['example2'])
    factors = np.concatenate(factor_list)

    za_by_factor = dict()
    zb_by_factor = dict()
    mean_by_factor = dict()
    var_by_factor = dict()
    score_by_factor = dict()
    individual_scores = dict()

    zall = np.concatenate([za,zb], 0)
    mean = np.mean(zall, 0, keepdims=True)

    var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
    for f in range(args.n_factors):
        if f != args.residual_index:
            indices = np.where(factors==f)[0]
            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
            var_by_factor[f] = np.var(np.concatenate((za_by_factor[f], zb_by_factor[f]), axis=0), 0, keepdims=True)
            # OG
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f]/var

            idv = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)/ (var_by_factor[f][0])
            individual_scores[f] = idv
        else:
            individual_scores[f] = np.ones(za_by_factor[0].shape[1])
            score_by_factor[f] = 1.0

    scores = np.array([score_by_factor[f] for f in range(args.n_factors)])
    idv_scores = np.array([individual_scores[f] for f in range(args.n_factors)])
    if args.joint_encoding:
        joint_dims = {'0': [], # motion
                      '1': [], # appearance
                      '2': [], # joint
                      '3': []} # none
        for i in range(idv_scores.shape[1]):
            if idv_scores[0,i] > float(args.joint_encoding_thresh) and idv_scores[1,i] > float(args.joint_encoding_thresh):
                joint_dims['2'].append(i)
            elif idv_scores[0,i] > float(args.joint_encoding_thresh) and idv_scores[1,i] < float(args.joint_encoding_thresh):
                joint_dims['0'].append(i)
            elif idv_scores[0,i] < float(args.joint_encoding_thresh) and idv_scores[1,i] > float(args.joint_encoding_thresh):
                joint_dims['1'].append(i)
            elif idv_scores[0, i] < float(args.joint_encoding_thresh) and idv_scores[1, i] < float(args.joint_encoding_thresh):
                joint_dims['3'].append(i)

        # plot bar plot
        num_motion = len(joint_dims['0'])
        num_app = len(joint_dims['1'])
        num_joint = len(joint_dims['2'])

        num_none = len(joint_dims['3'])

        fig = plt.figure()
        factors = ['Motion Dims', 'Appearance Dims', 'Joint Dims', 'None Dims']
        # num_dims = [num_motion, num_app, num_joint]
        num_dims = [num_motion, num_app, num_joint, num_none]
        plt.bar(factors, num_dims)
        plt.xlabel("Semantic Factors")
        plt.ylabel("No. of Encoding Channels")
        plt.title("Motion and App. Channels (Thresh = {})".format(args.joint_encoding_thresh))
        if 'slowfast' in args.model:
            plt.savefig('dim_outputs/joint_encode_plots_unique/thresh_' + str(args.joint_encoding_thresh) + '_' + args.model + \
                        '_' + args.dataset + '_' + args.path + '.png')
        elif 'rtnet' not in args.model and 'twostream_deeplabv3plus_resnet101' not in args.model and 'matnet' not in args.model:
            plt.savefig('dim_outputs/joint_encode_plots_unique/thresh_' + str(args.joint_encoding_thresh) + '_' + args.model +\
                        '_' + 'stg_' + str(args.stg) + '_' + args.dataset + '.png')
        else:
            with open('dim_outputs/vos_models/joint_encoding/%s/%s.pkl'%(args.model, args.stg), 'wb') as f:
                pickle.dump(joint_dims, f)
            with open('dim_outputs/vos_models/joint_encoding/%s/raw_%s.pkl'%(args.model, args.stg), 'wb') as f:
                pickle.dump(idv_scores, f)

    # SOFTMAX
    softmaxed = softmax_fn(scores, use_max=True)
    idv_softmax = []
    for i in range(idv_scores.shape[1]):
        idv_softmax.append(softmax_fn(idv_scores[:, i], use_max=True))

    idv_softmax = np.array(idv_softmax)

    dim = za.shape[1]
    dims = [int(s*dim) for s in softmaxed]
    dims[-1] = dim - sum(dims[:-1])
    dims_percent = dims.copy()
    for i in range(len(dims)):
        dims_percent[i] = round(100*(dims[i] / sum(dims)),1)

    if return_idv_scores:
        return dims, dims_percent, idv_softmax
    else:
        return dims, dims_percent
