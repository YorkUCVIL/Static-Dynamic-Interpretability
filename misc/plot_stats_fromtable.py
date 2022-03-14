import os
import numpy as np
import matplotlib.pyplot as plt

xaxis = 'mot'
yaxis = 'app'
streams = ['mot_stream', 'app_stream', 'sensor_fusion']
models = ['twostream_deeplabv3plus_resnet101', 'matnet', 'rtnet']
layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
seeds = [1, 2, 3, 4]

for model in models:
    for stream in streams:
        for layer in layers:
            if stream == 'sensor_fusion' and layer == 'conv1':
                continue

            if 'app' in stream:
                xaxis = 'app'
                yaxis = 'mot'
            else:
                xaxis = 'mot'
                yaxis = 'app'

            for seed in seeds:
                plt.figure()
                csv_file = 'dim_outputs/vos_models/stats_tables/%s/%s,%s_%d.csv'%(model, layer, stream, seed)
                if not os.path.exists(csv_file):
                    continue

                with open(csv_file, 'r') as f:
                    aas = []
                    bs = []

                    cs = []
                    ds = []
                    for line in f:
                        _, a, b, c, d = line.split(',')
                        aas.append(float(a))
                        bs.append(float(b))

                        cs.append(float(c))
                        ds.append(float(d))

                plt.title('Model: ' + model + ' ' + stream + ' Layer: ' + layer)
                if xaxis == 'mot':
                    plt.bar([1, 2], [np.mean(bs), np.mean(ds)], yerr = [np.std(bs), np.std(ds)], width=0.2, color='r', label='%s same'%yaxis)
                    plt.bar([1.2, 2.2], [np.mean(cs), np.mean(aas)], width=0.2, yerr=[np.std(cs), np.std(aas)], color='b', label='%s diff'%yaxis)
                else:
                    plt.bar([1, 2], [np.mean(aas), np.mean(ds)], yerr = [np.std(aas), np.std(ds)], width=0.2, color='r', label='%s same'%yaxis)
                    plt.bar([1.2, 2.2], [np.mean(cs), np.mean(bs)], width=0.2, yerr=[np.std(cs), np.std(bs)], color='b', label='%s diff'%yaxis)

                plt.xticks([1, 2], ['%s diff'%xaxis, '%s same'%xaxis])
                plt.legend()

                plt.savefig('plots_stats/%s_%s_%s_%d.png'%(model, stream, layer, seed))

