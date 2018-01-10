import json

import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('legend', fontsize=15)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('channels_reconstructed.pdf')

index=0
window_size=16
fsamp = 1

project_file_path = "/home/runge/openbci/git/OpenBCI_Python"
config_file = project_file_path + "/config/config.json"
raw_reconstructed_signals = pd.read_csv(project_file_path+"/build/dataset/train/result/raw_reconstructed_signals.csv")
raw_channels_data = pd.read_csv(project_file_path+"/build/dataset2017-5-6_0-0-33new_up.csv").ix[:,2:7].dropna()
# raw_channels_data = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/OpenBCI-RAW-right_strait_up_new.csv").ix[:,1:6].dropna()
channels_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]

graph_legend = []
handle_as=[]
labels_as=[]
start =1100
end = 0
num_ch = len(channels_names)

fig = plt.figure(figsize=(20, 14))
fig.subplots_adjust(hspace=.5)

with open(config_file) as config:
    config = json.load(config)
    config["train_dir_abs_location"] = project_file_path + "/build/dataset/train"

    for h in range(0, num_ch):
        preprocessor = PreProcessor(h, None, None, config)
        ax = plt.subplot(num_ch,1,h+1)

        # axes = plt.gca()
        # axes.set_ylim([0, 180])
        if(end==0):
            end = raw_channels_data.ix[:, h].shape[0]-1
        x = np.arange(start, end, 1)
        input_signal = raw_channels_data.ix[:, h][start * fsamp:end * fsamp]
        # l1 = ax.plot(input_signal, linewidth=1.0, label="raw signal")
        # graph_legend.append(l1)

        noise_reducer_signal = preprocessor.apply_noise_reducer_filer(input_signal)
        l2 = ax.plot(x, noise_reducer_signal, linewidth=3.0, label="noise_reducer_signal")
        graph_legend.append(l2)

        # normalize_signal = preprocessor.nomalize_signal(noise_reducer_signal)
        # l3 = ax.plot(x, normalize_signal, linewidth=3.0, label="normalize_signal")
        # graph_legend.append(l3)

        # reconstructed_signal = SingularSpectrumAnalysis(noise_reducer_signal, window_size, False).execute(1)
        # l4 = ax.plot(x,reconstructed_signal, linewidth=3.0, label='reconstructed signal with SSA')
        # graph_legend.append(l4)

        handles, labels = ax.get_legend_handles_labels()
        handle_as.append(handles)
        labels_as.append(labels)
        plt.title(channels_names[h])
        # leg = plt.legend(handles=handles, labels=labels)

    fig.legend(handles=handle_as[0], labels=labels_as[0])
    fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
    fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
    plt.show()


# pp.savefig(bbox_inches='tight')
# pp.close()


