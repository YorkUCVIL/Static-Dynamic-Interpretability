import numpy as np
import os

if __name__ == "__main__":
    nseeds = 4

    main_dir = 'dim_outputs/vos_models/results_final_4pairs/tatt_matnet'
    stage_map = {'conv1':0, 'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
    stage_map_reverse = {0:'conv1', 1:'layer1', 2:'layer2', 3:'layer3', 4:'layer4'}

    x = []
    y = []
    for i in range(nseeds):
        x.append({'app_stream': [], 'mot_stream': [], 'sensor_fusion': []})
        y.append({'app_stream': {'app': [], 'mot': [], 'residual': [], 'same': []},
            'mot_stream': {'app': [], 'mot': [], 'residual': [], 'same': []},
            'sensor_fusion': {'app': [], 'mot': [], 'residual': [], 'same': []}
                })

    seed_no = 0
    for element in sorted(os.listdir(main_dir)):
        if len(element.split('.')) < 2:
            continue
        if element.split('.')[1] != 'txt':
            continue

        with open(os.path.join(main_dir, element), 'r') as f:
            for line in f:
                tokens = line.split(":")
                stage = tokens[1].split(" ,")[0].strip()
                stage, stream = stage.split(",")
                x[seed_no][stream].append(stage_map[stage])

                dims = tokens[2].split("] [")[1].strip()[:-1]
                dims = dims.split(",")
                y[seed_no][stream]['mot'].append(float(dims[0]))
                y[seed_no][stream]['app'].append(float(dims[1]))
                y[seed_no][stream]['residual'].append(float(dims[2]))
                y[seed_no][stream]['same'].append(float(dims[3]))

        seed_no += 1

    for k, v in y[0].items():
        for stage in range(len(x[0][k])):
            current_app = []
            current_mot = []
            current_res = []
            current_same = []

            for i in range(nseeds):
                current_app.append(y[i][k]['app'][stage])
                current_mot.append(y[i][k]['mot'][stage])
                current_res.append(y[i][k]['residual'][stage])
                current_same.append(y[i][k]['same'][stage])
            print(k, ', Stage ', x[0][k][stage], ': App : ', np.mean(current_app), ' ', \
                    1.96*np.std(current_app)/np.sqrt(nseeds))
            print(k, ', Stage ', x[0][k][stage], ': Mot : ', np.mean(current_mot), ' ', \
                    1.96*np.std(current_mot)/np.sqrt(nseeds))
            print(k, ', Stage ', x[0][k][stage], ': Res : ', np.mean(current_res), ' ', \
                    1.96*np.std(current_res)/np.sqrt(nseeds))

            #print(k, ', Stage ', x[0][k][stage], ': Residual : ', 100 - (np.mean(current_app) + np.mean(current_mot)))

