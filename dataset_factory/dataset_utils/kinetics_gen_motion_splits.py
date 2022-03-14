import json
import random
import cv2
import glob
import os
from tqdm import tqdm
import shutil

kinetics_data_path = '/mnt/zeta_share_1/public_share/Datasets/kinetics_400/frames/'
data_list_csv = '/mnt/zeta_share_1/public_share/Datasets/kinetics_400/validate.csv'

# kin_temp_idx = [30, 34, 41, 45, 63, 75, 105, 147, 148, 152, 173, 177, 206, 222, 228, 230,
#                 235, 277, 295, 301, 302, 307, 308, 309, 310, 322, 325, 349, 357, 358, 376, 395]
# subset = 'temp'
# subset = 'static'
# subset = 'random'


subsets = ['temp', 'static', 'random']

for subset in subsets:
    print('Preparing --{}-- subset!'.format(subset))
    if subset == 'temp':
        kin_subset_names =  ["bouncing on trampoline", "breakdancing", "busking", "cartwheeling", "cleaning shoes",
                           "country line dancing", "drop kicking", "gymnastics tumbling", "hammer throw", "high kick",
                           "jumpstyle dancing", "kitesurfing", "parasailing", "playing cards", "playing cymbals",
                           "playing drums", "playing ice hockey", "robot dancing", "shining shoes", "shuffling cards",
                           "side kick", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry",
                           "skiing slalom", "snowboarding", "somersaulting", "tap dancing",
                           "throwing ball", "throwing discus", "vault", "wrestling"]
        save_dir = '/mnt/zeta_share_1/public_share/Datasets/kinetics_400/temp_stat_subsets/temp_val'
    elif subset == 'static':
        kin_subset_names = ['belly dancing', 'bending back', 'blasting sand', 'blowing nose', 'changing wheel',
         'clapping', 'curling hair', 'deadlifting', 'dining', 'doing aerobics', 'dribbling basketball', 'eating doughnuts',
         'filling eyebrows', 'getting a tattoo', 'laying bricks', 'long jump', 'lunge', 'making bed', 'moving furniture',
         'mowing lawn', 'peeling apples', 'playing badminton', 'playing controller', 'playing cricket', 'pull ups',
                            'riding camel', 'shot put', 'testifying', 'trimming trees', 'waxing eyebrows', 'yawning', 'yoga']
        save_dir = '/mnt/zeta_share_1/public_share/Datasets/kinetics_400/temp_stat_subsets/static_val'
    elif subset == 'random':
        kin_subset_names = ['arranging flowers', 'assembling computer', 'blowing out candles', 'bouncing on trampoline',
                            'busking', 'carrying baby', 'cleaning windows', 'cooking sausages', 'curling hair',
                            'dancing gangnam style', 'eating chips', 'eating doughnuts', 'egg hunting',
                            'feeding goats', 'gargling', 'grooming horse', 'hugging', 'making snowman', 'opening bottle',
                            'opening present', 'paragliding', 'parasailing', 'passing American football (in game)',
                            'peeling apples', 'playing cymbals', 'playing flute', 'presenting weather forecast',
                            'texting', 'tossing salad', 'waiting in line', 'watering plants', 'zumba']
        save_dir = '/mnt/zeta_share_1/public_share/Datasets/kinetics_400/temp_stat_subsets/random_val'



    with open(data_list_csv) as f:
        val_list = f.readlines()
        id_list = []
        empty_dir_list = []
        for x in val_list:
            line = x[:-1].split(',')
            lbl = line[0]
            if lbl in kin_subset_names:
                vid_id = kinetics_data_path + lbl + '/' + line[1]
                if os.path.exists(vid_id):
                    if len(glob.glob(vid_id + '/*')) > 0:
                        id_list.append(vid_id)
                    else:
                        print('Directory is empty: {}'.format(vid_id))
                        empty_dir_list.append(vid_id)

    for vid in tqdm(id_list):
        save_video_path = save_dir + '/' + vid.split('/')[-2]
        if not os.path.exists(save_video_path):
            os.mkdir(save_video_path)
        save_video_path += '/' + vid.split('/')[-1]
        if not os.path.exists(save_video_path):
            os.mkdir(save_video_path)
        # if 'crosscountry' in save_video_path and 'not' in save_video_path:
        vid_save = vid.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
        save_video_path = save_video_path.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
        command = 'cp -r {}/* {}'.format(vid_save, save_video_path)
        os.system(command)


print('Done!')
