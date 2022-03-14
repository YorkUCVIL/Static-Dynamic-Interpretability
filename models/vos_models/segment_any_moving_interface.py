import torch.nn as nn
import os
import numpy as np
from torchvision.transforms.functional import normalize
import cv2
from pathlib import Path
import logging
import subprocess

ROOT = Path(__file__).parent.joinpath('segment-any-moving/')

def resolve_path(x):
    if x.is_absolute():
        x = ROOT / x
    return x.resolve()

def subprocess_call(cmd, log=True, **kwargs):
    wd = os.getcwd()
    os.chdir(ROOT)

    cmd = [
        str(x) if not isinstance(x, Path) else str(resolve_path(x))
        for x in cmd
    ]
    if log:
        logging.debug('Command:\n%s', ' '.join(cmd).replace("--", "\\\n--"))
        if kwargs:
            logging.debug('subprocess.check_call kwargs:\n%s', kwargs)
    subprocess.check_call(cmd, **kwargs)
    os.chdir(wd)

def msg(message):
    logging.info(f'\n\n###\n{message}\n###\n')

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

    def denorm(self, tensor):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        _mean = -mean/std
        _std = 1/std
        return normalize(tensor, _mean, _std).permute(1,2,0)

    def forward(self, images, stages):
        """
        Infer through Segment Any Moving Joint Model
        """
        # Save Images and Flows
        original_tmp_dir = 'models/vos_models/segment-any-moving/.tmp_sam/'
        tmp_dir = original_tmp_dir.split('/')[-2]

        imgs_dir = os.path.join(original_tmp_dir, 'imgs')
        flows_dir = os.path.join(original_tmp_dir, 'flows')
        feats_dir = os.path.join(original_tmp_dir, 'feats')
        if not os.path.exists(original_tmp_dir):
            os.makedirs(imgs_dir)
            os.makedirs(flows_dir)
            os.makedirs(feats_dir)

        for batch_idx in range(images['Image'].shape[0]):
            img = self.denorm(images['Image'][batch_idx]) * 255
            flow = self.denorm(images['Flow'][batch_idx]) * 255
            cv2.imwrite(os.path.join(imgs_dir, '%05d.png'%batch_idx), img.cpu().numpy())
            cv2.imwrite(os.path.join(flows_dir, '%05d.png'%batch_idx), flow.cpu().numpy())
            with open(os.path.join(flows_dir, '%05d_magnitude_minmax.txt'%batch_idx), 'w') as f:
                f.write('0 1\n')
        # Infer
        subprocess_call([
                'python', 'release/custom/infer.py',
                '--frames-dir', Path(os.path.join(tmp_dir,'imgs')),
                '--flow-dir', Path(os.path.join(tmp_dir, 'flows')),
                '--model', "joint",
                '--config', Path("release/config.yaml"),
                '--output-dir', Path(os.path.join(tmp_dir, '.sam_results/')),
                '--stages', ':'.join(stages)
            ])

        # Load saved Features and COllate
        interm_feats = {}
        for fname in os.listdir(feats_dir):
            feats_dict = np.load(os.path.join(feats_dir, fname), allow_pickle=True).item()
            for key, value in feats_dict.items():
                if key not in interm_feats:
                    interm_feats[key] = []
                interm_feats[key].append(feats_dict[key])

        for key, value in interm_feats.items():
            interm_feats[key] = np.concatenate(interm_feats[key])

        # Cleanup
        os.system('rm -rf '+original_tmp_dir)
        return None, interm_feats
