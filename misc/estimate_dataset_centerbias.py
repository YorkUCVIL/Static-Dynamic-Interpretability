import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

prior_shape = (480, 854)#(513, 513)
saliency_prior = np.zeros(prior_shape, dtype=np.float)


def generate_prior(label_path):

    global saliency_prior

    label = cv2.imread(label_path)[:, :, 0]
    label[label > 0] = 1
    label = cv2.resize(label, prior_shape[::-1], interpolation=cv2.INTER_NEAREST)
    saliency_prior = saliency_prior + label


if __name__ == "__main__":
    dataset = 'taovos'

    roots = {'davis': "/local/riemann/home/msiam/DAVIS/Annotations/480p/",
             'taovos': "/local/riemann/home/msiam/TAO/annotations_vos/train/Annotations/",
             'imgnetvid': "/local/riemann/home/msiam/ImgnetVID/segmentation/ILSVRC2015/Data/VID/train/"}

    root = roots[dataset]
    regex = '*/*.png'
    if dataset == 'taovos':
        regex= '*/*/*.png'
    files = glob.glob(os.path.join(root, regex))
    print('Dataset Size ' + str(len(files)))

    for index in range(0, len(files)):
        image_id = files[index]
        print(image_id)
        generate_prior(image_id)


fig1 = plt.figure(1)
plt.figure(1)
plt.imshow(saliency_prior, cmap='gray')
fig1.tight_layout()
plt.axis('off')
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
#plt.show()
plt.savefig('saliency_prior_%s.png'%dataset, bbox_inches='tight', pad_inches=0)
