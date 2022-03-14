import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_tex(key, value, xaxis, first, out_plot):
    model_name = "FusionSeg"
    if key == "app_stream":
        stream = "Appearance Stream"
    else:
        stream = "Motion Stream"

    xs = xaxis[key]
    xs = [i+1 for i in xs]

    ys_motion = value['mot']
    ys_app = value['app']

    ymin = 10
    ymax = 50

    xmin = 0.5
    xmax = 5.5

    xticks = [x for x in range(int(xmin)+1, int(xmax)+1)]
    yticks = [y for y in range(ymin, ymax+5, 5)]

    with open(out_plot, 'a') as ftex:
        if first:
            ftex.write("\\begin{figure}[t]")

        ftex.write("\\begin{tikzpicture}\n \
                \\begin{axis}[\n \
                line width=1.0,\n \
                title={\\textbf{%s - %s}},\n \
                title style={at={(axis description cs:0.5,1.1)},anchor=north,font=\\normalsize},\n \
                xlabel={Network Stage},\n \
                ylabel={Bias},\n \
                xmin=%0.1f, xmax=%0.1f,\n \
                ymin=%0.1d, ymax=%0.1d,\n \
                xtick={"%(model_name, stream, xmin, xmax, ymin, ymax))

        for itr, xtick in enumerate(xticks):
            if itr < len(xticks)-1:
                ftex.write(str(xtick)+",")
            else:
                ftex.write(str(xtick)+"},\n")

        ftex.write("ytick={")
        for itr, ytick in enumerate(yticks):
            if itr < len(yticks)-1:
                ftex.write(str(ytick)+",")
            else:
                ftex.write(str(ytick)+"},\n")

        ftex.write("x tick label style={font=\\footnotesize},\n \
                y tick label style={font=\\footnotesize},\n \
                x label style={at={(axis description cs:0.5,-0.1)},anchor=north,font=\small},\n \
                y label style={at={(axis description cs:-0.08,.5)},anchor=south,font=\\normalsize},\n \
                width=6.5cm,\n \
                height=5cm,\n \
                ymajorgrids=true,\n \
                xmajorgrids=true,\n \
                major grid style={dotted,green!20!black},\n \
                legend style={\n \
                 nodes={scale=0.85, transform shape},\n \
                 cells={anchor=west},\n \
                 legend style={at={(1.2,1)},anchor=south,row sep=0.01pt}, font =\small},\n \
                 legend entries={[black]Motion,[black]Appearance},\n \
                legend to name=target_legend3,\n \
            ]\n")

        ftex.write("\\addplot[line width=1.6pt, mark size=1.1pt, color=orange, mark=*,]\n \
                coordinates {")
        for x, y in zip(xs, ys_motion):
            ftex.write("("+str(x)+","+str(y)+")")
        ftex.write("};\n")

        ftex.write("\\addplot[line width=1.6pt, mark size=1.1pt, color=cyan, mark=*,]\n \
                coordinates {")
        for x, y in zip(xs, ys_app):
            ftex.write("("+str(x)+","+str(y)+")")
        ftex.write("};\n")

        ftex.write("\end{axis}\n \
                \end{tikzpicture}\n \
                \hfill\n")

def plot(args):

    stage_map = {'conv1':0, 'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
    stage_map_reverse = {0:'conv1', 1:'layer1', 2:'layer2', 3:'layer3', 4:'layer4'}

    x = {'app_stream': [], 'mot_stream': []}
    y = {'app_stream': {'app': [], 'mot': []},
         'mot_stream': {'app': [], 'mot': []}}

    with open(args.in_file, 'r') as f:
        for line in f:
            tokens = line.split(":")
            stage = tokens[1].split(" ,")[0].strip()
            stage, stream = stage.split(",")
            if stream == "sensor_fusion":
                continue
            x[stream].append(stage_map[stage])

            dims = tokens[2].split("] [")[1].strip()[:-1]
            dims = dims.split(",")
            y[stream]['mot'].append(float(dims[0]))
            y[stream]['app'].append(float(dims[1]))

    fig = plt.figure(1)
    plt.ion()

    for it, (key, value) in enumerate(y.items()):
        if key == "app_stream":
            title = "Appearance Stream"
        elif key == "mot_stream":
            title = "Motion Stream"

        if args.matplot:
            plt.figure(1)
            plt.title("Stage-wise Dimensionality Estimation "+ title)
            plt.xlabel("Stages")
            plt.ylabel("Dimensionality")

            x_current = x[key]
            x_labels = [stage_map_reverse[i] for i in x_current]
            x_current = [i+1 for i in x_current]
            plt.xticks(x_current, x_labels)

            plt.plot(x_current, value['app'], "r-", label="Appearance")
            plt.plot(x_current, value['mot'], "b-", label="Motion")

            plt.legend()
            plt.savefig(args.out_plot%key)
            plt.waitforbuttonpress()
            fig.clear()
        else:
            # Generate Latex Code for Plots
            generate_tex(key, value, x, (it==0), args.out_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plotting Stagewise Dimensionality")
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_plot", type=str)
    parser.add_argument("--matplot", action="store_true")
    args = parser.parse_args()
    plot(args)

