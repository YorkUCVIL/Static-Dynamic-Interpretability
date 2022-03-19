import glob
import random
import cv2
import numpy as np
import torch.utils.data as data
import torch
import json
import math
import os
from itertools import cycle, islice
from PIL import Image
import sys
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from models.ar_models.pytorchvideo.pytorchvideo.transforms.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import copy

#
def get_stylized_activitynet_data(args):
    with open(os.path.join(args.stylized_data_dir, 'activitynet_v13.json')) as file:
        json_file = json.load(file)
    database = json_file['database']
    taxonomy = json_file['taxonomy']

    # prepare class dictionary with lists
    data_by_class = {}
    sport_data_by_class = {}
    sport_classes = []
    sport_parents = ['Participating in Sports, Exercise, or Recreation',
                     'Working out',
                     'Playing sports']
    for action in taxonomy:
        # if action['nodeName'].lower() != action['parentName'].lower():
        data_by_class[action['nodeName']] = []
        if action['parentName'] in sport_parents:
            sport_classes.append(action['nodeName'])

    skipped_videos = []
    data = []
    for video_id in database.keys():
        skip = False
        if database[video_id]['subset'] == args.subset:
            # create data sample
            cls = database[video_id]['annotations'][0]['label']
            num_frames = database[video_id]['num_frames']
            if num_frames < 32:
                skip = True
            # retrieve frames (5 fps)
            annotations = []
            for ann in database[video_id]['annotations']:
                frame_start = math.floor(ann['segment'][0]*5)
                frame_end = math.floor(ann['segment'][1]*5)
                if frame_end > num_frames:
                    frame_end = num_frames
                if frame_end - frame_start < 16:
                    skip = True
                annotations.append([frame_start, frame_end])

            # skip short videos or with segments that are too short
            if skip:
                skipped_videos.append(video_id)
                continue

            if video_id == 'm9CbLJdYqHw':
                print(1)

            sample = {
                'video_id': video_id,
                'cls': database[video_id]['annotations'][0]['label'],
                'num_frames': num_frames,
                'annotations': annotations
            }
                # create class sample


            data_by_class[cls].append(sample)
            data.append(sample)

    labels = list(data_by_class.keys())
    for label in labels:
        if len(data_by_class[label]) == 0:
            del data_by_class[label]
    print('A total of {} videos skipped!'.format(len(skipped_videos)))
    return data, data_by_class


def get_data_transforms(config, side_size = 256,mean = [0.45, 0.45, 0.45],std = [0.225, 0.225, 0.225],
                        crop_size = 256, frames_per_second = 30):
    if 'slowfast' in config.model:
        # SLOW
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.slow_num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )

        # FAST
        transform_fast = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.fast_num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        clip_duration = (config.fast_num_frames * config.fast_sampling_rate) / frames_per_second
    elif 'x3d' in config.model:
        num_frames = config.num_frames
        sampling_rate = config.sampling_rate
        crop_size = config.image_size
        side_size = config.scale_size
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        transform_fast = None
    elif 'mvit' in config.model:
        num_frames = config.num_frames
        crop_size = config.image_size
        sampling_rate = config.sampling_rate
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        transform_fast = None
    else:
        num_frames = config.num_frames
        sampling_rate = config.sampling_rate
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        transform_fast = None


    return transform, transform_fast, clip_duration


### stylized activitynet ###
class StylizedActivityNet(data.Dataset):
    def __init__(self, config):
        # need to replace this with datalist of all images, with corresponding styles and class labels
        self.config = config
        self.data_root = config.stylized_data_dir + '/style_'
        self.styles = [int(style) for style in config.styles.split(',')]
        self.data, self.data_by_class = get_stylized_activitynet_data(config)
        self.num_textures = len(self.styles)
        self.app_shuffle = config.app_shuffle
        self.n_factors = config.n_factors-2
        self.prng = np.random.RandomState(1)
        self.transform, self.transform_fast, self.clip_duration = get_data_transforms(config=config)

    def get_video_as_tensor(self, data_dict, style, app_normal=False):
        if self.config.use_normal_app:
            if app_normal:
                data_path = self.normal_data_root + 'v_' + data_dict['video_id']
            else:
                data_path = self.data_root + str(style) + '/v_' + data_dict['video_id']
        else:
            data_path = self.data_root + str(style) + '/v_' + data_dict['video_id']
        # choose starting frame
        # starting_frame = random.randint(0, data_dict['num_frames'] - self.config.n_frames)
        # num_action_instances = len(data_dict['annotations'])-1
        # instance = random.randint(0, num_action_instances)

        # choose first instance to keep things consistent
        instance = 0
        starting_frame = data_dict['annotations'][instance][0]
        end_frame = data_dict['annotations'][instance][1]
        frames = list(range(starting_frame+1, end_frame+1))
        # repeat video until n_frames length
        frames = list(islice(cycle(frames), self.config.n_sample_frames))
        video = []
        for i in frames:
            img_path = data_path + '/' + '{:06d}.jpg'.format(i)
            img = torch.tensor(np.array(Image.open(img_path))).permute(2,0,1)
            video.append(img)
        if len(video) == 0:
            print(data_dict['video_id'])
        return torch.stack(video, 1), starting_frame



    def __getitem__(self, i):  # motion and appearance
        # sample video + style
        # if motion difference, randomly choose new video id but keep style
        # elif appearance difference, choose new style and replace last element in data string

        # START HERE
        data1 = self.data[i]
        style1 = random.choice(self.styles)
        factor = random.randint(0, self.n_factors)

        if self.app_shuffle and self.config.use_normal_app and factor == 1:
            video1, start_frame = self.get_video_as_tensor(data1, style1, app_normal=True)
        else:
            video1, start_frame = self.get_video_as_tensor(data1, style1)
        cls1 = data1['cls']

        if factor == 0:
            # same motion, different appearance
            # remove style used already and choose another style
            list_possible_styles = self.styles.copy()
            list_possible_styles.remove(style1)
            style2 = random.choice(list_possible_styles)
            # grab same video with different style
            if self.config.m_same:
                video2 = self.get_video_as_tensor(data1, style2)
            else:
                list_possible_motions = self.data_by_class[cls1]
                list_possible_motions.remove(data1)
                data2 = random.choice(list_possible_motions)
                video2 = self.get_video_as_tensor(data2, style2)
        elif factor == 1:
            # same appearance, different motion
            # remove current action
            if self.app_shuffle:
                pass
            else:
                list_other_motions = list(self.data_by_class.keys())
                list_other_motions.remove(cls1)
                # choose new action class
                new_motion = random.choice(list_other_motions)
                #choose new video from that action
                data2 = random.choice(self.data_by_class[new_motion])
                # open video with same style
                video2 = self.get_video_as_tensor(data2, style1)
        elif factor == 2:
            # different appearance different motion
            list_possible_styles = self.styles.copy()
            list_possible_styles.remove(style1)
            style2 = random.choice(list_possible_styles)
            # grab same video with different style
            if self.config.m_same:
                video2 = self.get_video_as_tensor(data1, style2)
            else:
                list_possible_motions = self.data_by_class[cls1]
                list_possible_motions.remove(data1)
                data2 = random.choice(list_possible_motions)
                video2 = self.get_video_as_tensor(data2, style2)

        elif factor == 3:
            # same appearance same motion
            print('error3')
            video2 = copy.deepcopy(video1)

        # pre-process both videos
        if 'slowfast' in self.config.model:
            video1 = [self.transform(video1), self.transform_fast(video1)]
            if self.app_shuffle and factor == 1:
                rand_perm1 = torch.randperm(video1[0].shape[1])
                rand_perm2 = torch.randperm(video1[1].shape[1])
                video2 = [video1[0][:,rand_perm1], video1[1][:,rand_perm2]]
            elif self.app_shuffle and factor == 2:
                video2 = [self.transform(video2), self.transform_fast(video2)]
                rand_perm1 = torch.randperm(video2[0].shape[1])
                rand_perm2 = torch.randperm(video2[1].shape[1])

                video2 = [video2[0][:,rand_perm1], video2[1][:,rand_perm2]]
            else:
                video2 = [self.transform(video2), self.transform_fast(video2)]
        else:
            video1 = self.transform(video1)
            if self.app_shuffle and factor == 1:
                rand_perm = torch.randperm(video1.shape[1])
                video2 = video1[:, rand_perm]
            elif self.app_shuffle and factor == 2:
                rand_perm = torch.randperm(video2.shape[1])
                video2 = self.transform(video2[:, rand_perm])
            else:
                video2 = self.transform(video2)

        return factor, video1, video2

    def __len__(self):
        return len(self.data)*self.nrep
