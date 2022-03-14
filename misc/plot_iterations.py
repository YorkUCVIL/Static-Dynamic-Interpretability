import os
import matplotlib.pyplot as plt
import argparse

def plot(args):

    x = {'app_stream': [], 'mot_stream': [], 'sensor_fusion_layer1': [], 'sensor_fusion_layer4': []}
    y = {'app_stream': {'app': [], 'mot': []},
         'mot_stream': {'app': [], 'mot': []},
         'sensor_fusion_layer1': {'app': [], 'mot': []},
         'sensor_fusion_layer4': {'app': [], 'mot': []}}

    for file_ in sorted(os.listdir(args.in_dir)):
        iter_file = os.path.join(args.in_dir, file_)
        try:
            itr = int(file_.split('_')[-1].split('.')[0])
        except:
            itr = int(file_.split('_')[-3])

        with open(iter_file, 'r') as f:
            for line in f:
                tokens = line.split(":")
                stage = tokens[1].split(" ,")[0].strip()
                stage, stream = stage.split(",")
                if stream != "sensor_fusion":
                    continue
                x[stream+'_'+stage].append(itr)

                dims = tokens[2].split("] [")[1].strip()[:-1]
                dims = dims.split(",")
                y[stream+'_'+stage]['mot'].append(float(dims[0]))
                y[stream+'_'+stage]['app'].append(float(dims[1]))

    fig = plt.figure(1)
    plt.ion()

    for key, value in y.items():
        if key == "app_stream":
            title = "Appearance Stream"
            continue
        elif key == "mot_stream":
            title = "Motion Stream"
            continue
        elif key == "sensor_fusion_layer1":
            title = "Sensor Fusion Layer1"
        elif key == "sensor_fusion_layer4":
            title = "Sensor Fusion Layer4"

        plt.figure(1)
        plt.title("Dimensionality Estimation w.r.t Training Iterations "+ title)
        plt.xlabel("Stages")
        plt.ylabel("Dimensionality")

        x_current = x[key]
        plt.plot(x_current, value['app'], "r-", label="Appearance")
        plt.plot(x_current, value['mot'], "b-", label="Motion")

        plt.legend()
        plt.savefig(args.out_plot%key)
#        plt.waitforbuttonpress()
        fig.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plotting Stagewise Dimensionality")
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_plot", type=str)
    args = parser.parse_args()
    plot(args)

