import glob
import json
import sys
import os

data_path = '/mnt/zeta_share_1/public_share/Datasets/DTDB/DTDB/*'


# file locations
data_root = '/media/ssd4/m3kowal/DTDB/DTDB/'
file_path = '/mnt/zeta_share_1/public_share/Datasets/DTDB/DTDB/Conversion_scripts/'
dyn_files = ['Dyn2App_correspondence_TEST.csv', 'Dyn2App_correspondence_TRAIN.csv']
app_files = ['App2Dyn_correspondence_TEST.csv', 'App2Dyn_correspondence_TRAIN.csv']

# create list of unique video id's

vid_dict = {} # each id with both types of labels has a dyn value and an app value
# dynamics first
for file in dyn_files:
    with open(file_path + file) as f:
        lines = f.readlines()
        for line in lines:
            video_id = line.split(', ')[0]
            dyn_label = line.split(', ')[1][:-1]
            if 'TEST' in file:
                dyn_subset = 'test'
                dyn_subset_cap = 'TEST'
            else:
                dyn_subset = 'train'
                dyn_subset_cap = 'TRAIN'

            cls = dyn_label.split('_g')[0]
            vid_path = data_root + 'BY_DYNAMIC_FINAL/' + dyn_subset_cap + '/' + cls + '/' + dyn_label
            if not os.path.exists(vid_path):
                continue
            vid_dict[video_id] = {'dynamic': dyn_label, 'dyn_subset': dyn_subset}

#app second
for file in app_files:
    with open(file_path + file) as f:
        lines = f.readlines()
        for line in lines:
            video_id = line.split(', ')[0]
            app_label = line.split(', ')[1][:-1]
            if 'TEST' in file:
                app_subset = 'test'
                app_subset_cap = 'TEST'
            else:
                app_subset = 'train'
                app_subset_cap = 'TRAIN'

            cls = app_label.split('_g')[0]

            if cls == 'Sliding':
                cls = 'Sliding_gate'

            vid_path = data_root + 'BY_APPEARANCE_FINAL/' + app_subset_cap + '/' + cls + '/' + app_label


            if not os.path.exists(vid_path):
                continue

            vid_dict[video_id]['appearance'] = app_label
            vid_dict[video_id]['app_subset'] = app_subset

vid_dict_copy = vid_dict.copy()
for vid in vid_dict_copy:
    if len(vid_dict[vid]) < 3:
        del vid_dict[vid]

for vid in vid_dict:
    if not len(vid_dict[vid]) == 4:
        sys.exit()


with open('/mnt/zeta_share_1/public_share/Datasets/DTDB/DTDB/app_dyn_correspondance.json', 'w') as f:
    json.dump(vid_dict, f)
with open('/media/ssd4/m3kowal/DTDB/DTDB/app_dyn_correspondance.json', 'w') as f:
    json.dump(vid_dict, f)
