import torch
if torch.__version__ == "0.4.0":
    OLDTORCH = True
else:
    OLDTORCH = False
import copy
import random
import torch.utils.data as data
import json
import numpy as np
import os
from PIL import Image
import torch.utils.data as data
import torch
from dataset_factory.vos_transforms import ExtToTensor, ExtNormalize, ExtCompose, \
            ExtResize, ExtCenterCrop, ExtFlowColorJitter

class PairedConcatDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) == 2

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        assert len(self.datasets[0]) == len(self.datasets[1])
        return len(self.datasets[0])

class StylizedDAVIS(data.Dataset):
    def __init__(self, config, debug=False):
        stylized_root, root, split, transform, \
            sampling_method, noisy_transform, nreps = self._load_data_cfgs(config.cfg_file)
        self.transform = transform
        self.noisy_transform = noisy_transform
        self.nreps = nreps
        self.debug = debug

        if hasattr(config, 'fwbwflow') and config.fwbwflow:
            self.seqcls = StylizedDAVISFWBWSequence
        else:
            self.seqcls = StylizedDAVISSequence
        # Sampling Appearance Pair:
        # (1) noise_flow: Use same style and sequence, add noise on flow
        # (2) diff_sq: Use same style, different sequence
        self.sampling_method = sampling_method

        self.styles = [int(style) for style in config.styles.split(',')]
        self.n_factors = config.n_factors
        if config.residual_index != -1:
            self.n_factors = self.n_factors - 1
        self.current_factor = 0

        self.prng = np.random.RandomState(1)
        self.file_names = self._load_filenames(root, split)

        self.root_prefix = os.path.join(root, 'JPEGImages/480p')
        self.stylized_root_prefix = os.path.join(stylized_root, 'JPEGImages/')

        self.style_map = {1: 'Lynx', 2: 'Maruska640', 3: 'Zuzka1', 4: 'Zuzka2'}
        self.flow_sixdigit = False

    def _load_data_cfgs(self, data_cfg):
        with open(data_cfg, 'r') as f:
            data_cfg = json.load(f)

        root = data_cfg['original_data_root']
        stylized_root = data_cfg['stylized_data_root']
        split = data_cfg['split']
        sampling_method = data_cfg['sampling_method']
        nreps = data_cfg['nreps']

        transform = ExtCompose([ExtResize(data_cfg['crop_size']),
                                ExtCenterCrop(data_cfg['crop_size']),
                                ExtToTensor(),
                                ExtNormalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),])
        noisy_transform = None
        if sampling_method == "noisy_flow":
            noisy_transform = ExtCompose([ExtResize(data_cfg['crop_size']),
                        ExtCenterCrop(data_cfg['crop_size']),
                        ExtFlowColorJitter(),
                        ExtToTensor(),
                        ExtNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),])

        return stylized_root, root, split, transform, \
                    sampling_method, noisy_transform, nreps


    def _load_filenames(self, root, split):
        image_set_file = os.path.join(root, 'ImageSets/480p/'+split+'.txt')
        fil = open(image_set_file, 'r')

        filenames = {}
        for line in fil:
            sq_name = line.split()[0][1:].split('/')[-2]
            if sq_name not in filenames:
                filenames[sq_name] = []

            filenames[sq_name].append(os.path.join(root, line.split()[0][1:] ))

        return filenames

    def __getitem__(self, index):
        selected_sq = list(self.file_names.keys())[index%len(self.file_names)]
        # Generate random factor
        self.current_factor = (self.current_factor + 1) % self.n_factors
        factor = self.current_factor
        style1 = random.choice(self.styles)
        style1_name = self.style_map[style1]

        scope = 10
        other_idx = np.random.randint(-scope, scope)

        stylized_root_prefix = os.path.join(self.stylized_root_prefix, style1_name)
        video1 = self.seqcls(self.file_names[selected_sq], self.transform,
                                       self.root_prefix, stylized_root_prefix,
                                       flow_sixdigit=self.flow_sixdigit,
                                       other=other_idx)

        # Pick pairs according to factor
        if factor == 0:
            # Same motion different appearance
            rest_styles = list(set(self.styles) - set([style1]))
            style2_name = self.style_map[random.choice(rest_styles)]
            stylized_root_prefix = os.path.join(self.stylized_root_prefix, style2_name)
            video2 = self.seqcls(self.file_names[selected_sq], self.transform,
                                           self.root_prefix, stylized_root_prefix,
                                           flow_sixdigit=self.flow_sixdigit,
                                           other=other_idx)
        elif factor == 1:
            style2_name = style1_name
            # same appearance different motion
            if self.sampling_method == "diff_sq":
                sq2 = random.choice(list(set(self.file_names.keys()) - set(selected_sq)))
                video2 = selfseqcls(self.file_names[sq2], self.transform,
                                               self.root_prefix, stylized_root_prefix,
                                               flow_sixdigit=self.flow_sixdigit,
                                               other=other_idx)

            elif self.sampling_method == "noisy_flow":
                video2 = self.seqcls(self.file_names[selected_sq],
                                       self.noisy_transform, self.root_prefix,
                                       stylized_root_prefix,
                                       flow_sixdigit=self.flow_sixdigit,
                                       other=other_idx)
            else:
                raise NotImplementedError
        elif factor == 2:
            rest_styles = list(set(self.styles) - set([style1]))
            style2_name = self.style_map[random.choice(rest_styles)]
            stylized_root_prefix = os.path.join(self.stylized_root_prefix, style2_name)
            video2 = self.seqcls(self.file_names[selected_sq], self.noisy_transform,
                                 self.root_prefix, stylized_root_prefix,
                                 flow_sixdigit=self.flow_sixdigit,
                                 other=other_idx)

        elif factor == 3:
            style2_name = style1_name
            video2 = copy.deepcopy(video1)


        if self.debug:
            return factor, video1, video2, selected_sq, style1_name, style2_name
        else:
            return factor, video1, video2

    def __len__(self):
        # Number of sequneces
        #import pdb; pdb.set_trace()
        return len(self.file_names) * self.nreps

class StylizedDAVISSequence(data.Dataset):
    def __init__(self, filenames, transform, original_prefix, stylized_prefix,
                 flow_sixdigit=True, other=-1):

        self.filenames = sorted(filenames[:-1])
        self.transform = transform

        self.original_prefix = original_prefix
        self.stylized_prefix = stylized_prefix
        self.flow_sixdigit = flow_sixdigit

    def __getitem__(self, index):
        image_path = self.filenames[index]
        stylized_image_path = image_path.replace(self.original_prefix, self.stylized_prefix)
        img = Image.open(stylized_image_path)

        annot_path = image_path.replace('JPEGImages', 'Annotations').replace('jpg', 'png')
        annot = Image.open(annot_path)

        # TODO: Generate Flow on the stylized sequences and read from
        flow_path = image_path.replace('JPEGImages/480p', 'FlowImages_gap1').replace('jpg', 'png')

        flow = Image.open(flow_path)
        flow = flow.resize(img.size)

        img, annot, flow = self.transform(img, annot, flow)
        if OLDTORCH:
            annot = annot / torch.tensor(255).byte()
            if len(annot.shape) > 2:
                annot = annot[:, :, 0]
        else:
            annot = annot // 255
            if annot.ndim > 2:
                annot = annot[:, :, 0]

        return {'Image': img, 'Flow': flow, 'Annot':annot}

    def __len__(self):
        # Number of frames in sequence
        return len(self.filenames)

class StylizedDAVISFWBWSequence(StylizedDAVISSequence):
    def __init__(self, filenames, transform, original_prefix, stylized_prefix,
                 flow_sixdigit=True, other=-1):
        super(StylizedDAVISFWBWSequence, self).__init__(filenames=filenames, transform=transform,
                                                        original_prefix=original_prefix, stylized_prefix=stylized_prefix,
                                                        flow_sixdigit=flow_sixdigit)
        self.filenames = self.filenames[1:] # Avoid ones without fw/bw flow
        self.other = other

    def _read_image_label_flow(self, image_path):
        stylized_image_path = image_path.replace(self.original_prefix, self.stylized_prefix)
        img = Image.open(stylized_image_path)

        annot_path = image_path.replace('JPEGImages', 'Annotations').replace('jpg', 'png')
        annot = Image.open(annot_path)

        fwflow_path = image_path.replace('JPEGImages/480p', 'FlowImages_gap1').replace('jpg', 'png')
        bwflow_path = image_path.replace('JPEGImages/480p', 'FlowImages_gap-1').replace('jpg', 'png')

        fwflow = Image.open(fwflow_path)
        bwflow = Image.open(bwflow_path)

        img, annot, flow = self.transform(img, annot, {'fw': fwflow, 'bw': bwflow})
        annot = annot // 255
        if annot.ndim > 2:
            annot = annot[:, :, 0]

        flow = torch.cat((flow['fw'], flow['bw']), 0)
        return img, annot, flow

    def __getitem__(self, index):
        image_path = self.filenames[index]
        img, annot, flow = self._read_image_label_flow(image_path)

        image_path2 = self.filenames[(index + self.other)%len(self.filenames)]
        img2, annot2, flow2 = self._read_image_label_flow(image_path2)

        img = torch.stack((img, img2))
        annot = torch.stack((annot, annot2))
        flow = torch.stack((flow, flow2))

        return {'Image': img, 'Flow': flow, 'Annot':annot}
