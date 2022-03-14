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


class StylizedDAVISCinematograms(StylizedDAVIS):
    def __init__(self, config, debug=False):
        self.cinematogram_length = 10
        super(StylizedDAVISCinematograms, self).__init__(config, debug)
        self.flow_sixdigit = False
        self.styles = self.styles[:-1]

    def _load_filenames(self, root, split):
        image_set_file = os.path.join(root, 'ImageSets/480p/'+split+'.txt')
        fil = open(image_set_file, 'r')

        filenames = {}
        for line in fil:
            line = line.strip()

            initial_fname = os.path.join(root, line.split()[0][1:]).split('.')[0]
            if not os.path.exists(initial_fname):
                continue

            tokens = line.split()[0][1:].split('/')
            sq_name = tokens[-2]
            current_fname = tokens[-1].split('.')[0]

            sq_name = sq_name + '_' + current_fname

            if sq_name not in filenames:
                filenames[sq_name] = []

            fname = line.split()[0][1:]
            for i in range(self.cinematogram_length):
                old_fname = fname.split('/')[-1]
                new_fname = fname.replace('/'+old_fname, '/'+current_fname+'/%05d.jpg'%i)
                img_path = os.path.join(root, new_fname)
                flow_path = img_path.replace('JPEGImages/480p', 'OpticalFlow_Real').replace('jpg', 'png')
                if not os.path.exists(flow_path):
                    continue

                filenames[sq_name].append(img_path)

        return filenames
