import json
import random
import cv2
import glob
import os
from tqdm import tqdm

# TO GENERATE SAMPLES
# random.seed(10)
# val_data_path = "/mnt/zeta_share_1/public_share/Datasets/ssv2/something-something-v2-validation.json"
# with open(val_data_path) as file:
#     json_file = json.load(file)
# samples = random.sample(json_file, 5000)
# with open('/mnt/zeta_share_1/public_share/Datasets/ssv2/ssv2_val5k.json', 'w') as outfile:
#     json.dump(samples, outfile)





def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      success,image = vidcap.read()
      if not success:
          return
      cv2.imwrite(pathOut + "{:06d}.jpg".format(count), image)     # save frame as JPEG file
      count += 1

if __name__=="__main__":

    # AFTER SAMPLES ARE GENERATED
    val_data_path = "/mnt/zeta_share_1/public_share/Datasets/ssv2/ssv2_val5k.json"
    with open(val_data_path) as file:
        sample_videos = json.load(file)

    save_root = '/mnt/zeta_share_1/public_share/Datasets/ssv2/val_5k/'
    video_root = '/mnt/zeta_share_1/public_share/Datasets/ssv2/20bn-something-something-v2/'

    for video in tqdm(sample_videos):
        path_in = video_root + video['id'] + '.webm'
        path_out = save_root + video['id'] + '/'
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        extractImages(path_in, path_out)


# /mnt/zeta_share_1/public_share/Datasets/ssv2/val_5k/76281/000033.jpg