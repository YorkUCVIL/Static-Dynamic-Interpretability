import pickle
import matplotlib.pyplot as plt
import os
import sys

def threshold_dims(idv_scores, threshold=0.5):
    joint_dims = {'0': [], '1': [], '2': [], '3': []}
    for i in range(idv_scores.shape[1]):
        if idv_scores[0,i] > threshold and idv_scores[1,i] > threshold:
            joint_dims['2'].append(i)
        elif idv_scores[0,i] > threshold and idv_scores[1,i] < threshold:
            joint_dims['0'].append(i)
        elif idv_scores[0,i] < threshold and idv_scores[1,i] > threshold:
            joint_dims['1'].append(i)
        elif idv_scores[0, i] < threshold and idv_scores[1, i] < threshold:
            joint_dims['3'].append(i)
    return joint_dims

def plot(fname, outfile, use_raw=False, threshold=0.5):

    with open(fname, 'rb') as f:
        joint_dims = pickle.load(f)

    if use_raw:
        joint_dims = threshold_dims(joint_dims, threshold=threshold)

    total_neurons = 0
    for k, v in joint_dims.items():
        total_neurons += len(v)
    total_neurons = float(total_neurons)

    num_motion = len(joint_dims['0']) / total_neurons
    num_app = len(joint_dims['1']) / total_neurons
    num_joint = len(joint_dims['2']) / total_neurons
    num_none = len(joint_dims['3']) / total_neurons

#    fig = plt.figure()
    factors = ['Dynamic Dims', 'Static Dims', 'Joint Dims', 'None Dims']
    num_dims = [num_motion, num_app, num_joint, num_none]
    print(fname)
    print(factors, ' ', [round(n*100, 1) for n in num_dims])
    plt.bar(factors, num_dims)
    plt.xlabel("Semantic Factors")
    plt.ylabel("No. of Encoding Channels")
    plt.title("Dynamic and Static Channels (Thresh = %0.1f)"%threshold)
    plt.savefig(outfile)

if __name__ == "__main__":
    use_raw = True
    threshold = 0.5

    main_dir = 'dim_outputs/vos_models/joint_encoding/matnet_fusion_type_gated/'
    if len(sys.argv) == 2:
        main_dir = sys.argv[1]

    out_dir = main_dir.replace('joint_encoding', 'joint_encoding_plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname in os.listdir(main_dir):
        if 'sensor_fusion' in fname:
            if use_raw and 'raw' in fname:
                plot(os.path.join(main_dir, fname), os.path.join(out_dir, fname.replace('pkl', 'png')),
                     use_raw=use_raw, threshold=threshold)
            elif not use_raw and not 'raw' in fname:
                plot(os.path.join(main_dir, fname), os.path.join(out_dir, fname.replace('pkl', 'png')),
                     use_raw=use_raw, threhsold=threshold)


