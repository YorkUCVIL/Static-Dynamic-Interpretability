import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

#main_dir = 'matnet_hist'
main_dir = 'dim_outputs/vos_models/idv_scores/twostream_deeplabv3plus_resnet101/'
for fname in os.listdir(main_dir):
#    if 'idv_scores' not in fname:
#        continue

    histograms = []
#    stage = fname.replace('idv_scores_', '').split('.')[0]
    stage = fname.split('.')[0]

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    plt.ion()

    with open(os.path.join(main_dir, fname), 'rb') as f:
        output_dict = pickle.load(f)
        output_dict = np.stack(output_dict)
        plt.figure(1)
        plt.hist(output_dict[:, 0] * 100)
        plt.xlabel('Bias %')
        plt.ylabel('# Neurons in Bias Range')
        plt.title('FusionSeg Sensor Fusion Layer4 Motion Histogram')
        plt.savefig('%s_mot_hist.png'%stage)

        plt.figure(2)
        plt.hist(output_dict[:, 1] * 100)
        plt.xlabel('Bias %')
        plt.ylabel('# Neurons in Bias Range')
        plt.title('FusionSeg Sensor Fusion Layer4 Appearance Histogram')
        plt.savefig('%s_app_hist.png'%stage)

        fig1.clear()
        fig2.clear()

