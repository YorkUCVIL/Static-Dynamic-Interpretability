import random
import torch.utils.data as data
import json
import numpy as np
import os
from PIL import Image
import torch.utils.data as data
import torch
from dataset_factory.StylizedDAVIS import StylizedDAVIS
from dataset_factory.vos_transforms import ExtToTensor, ExtNormalize, ExtCompose, \
            ExtResize, ExtCenterCrop, ExtFlowColorJitter


class StylizedDAVISStereograms(StylizedDAVIS):
    def __init__(self, config, debug=False):
        self.stereogram_length = 10
        super(StylizedDAVISStereograms, self).__init__(config, debug)
        self.flow_sixdigit = False

    def _load_filenames(self, root, split):
        image_set_file = os.path.join(root, 'ImageSets/480p/'+split+'.txt')
        fil = open(image_set_file, 'r')

        filenames = {}
        for line in fil:
            line = line.strip()

            tokens = line.split()[0][1:].split('/')

            sq_name = tokens[-2]
            current_fname = tokens[-1].split('.')[0]

            sq_name = sq_name + '_' + current_fname

            if sq_name not in filenames:
                filenames[sq_name] = []

            fname = line.split()[0][1:]
            for i in range(self.stereogram_length):
                old_fname = fname.split('/')[-1]
                new_fname = fname.replace('/'+old_fname, '/'+current_fname+'/%05d.jpg'%i)
                filenames[sq_name].append(os.path.join(root, new_fname))

        return filenames
