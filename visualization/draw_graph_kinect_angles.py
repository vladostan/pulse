import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.ssa import SingularSpectrumAnalysis

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('kinect_angles_reconstructed.pdf')

index=0
window_size=128
fsamp = 1
# kinect__angles = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_kinect__angles_.csv")
raw_kinect_angle_data = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset2017-5-5_23-55-32new_bycept.csv").ix[:,10:13].dropna()
# raw_kinect_angle_data = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_kinect__angles_.csv").dropna()
raw_kinect_angle_data = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_kinect__angles_.csv").dropna()

angle_names = ["wrist", "elbow", "shoulder"]
graph_legend = []
handle_as=[]
labels_as=[]
start = 2100
end = 2600
num_ch = 3


fig = plt.figure(figsize=(20, 14))
fig.subplots_adjust(hspace=.5)
for h in range(0, num_ch):
    ax = plt.subplot(num_ch,1,h+1)
    axes = plt.gca()
    # axes.set_ylim([0, 180])
    if(end==0):
        end = raw_kinect_angle_data.ix[:, h].shape[0]-1

    input_signal =  raw_kinect_angle_data.ix[:, h][start * fsamp:end * fsamp]
    x = np.arange(start,end,1)
    # mean = np.mean(input_signal, axis=0)
    # input_signal -= mean
    # input_signal=input_signal / np.std(input_signal, axis=0)
    l1 = ax.plot(x,input_signal, linewidth=3.0, label="raw signal")
    graph_legend.append(l1)
    # reconstructed_signal = SingularSpectrumAnalysis(input_signal, window_size, False).execute(1)
    # l2 = ax.plot(x,reconstructed_signal, linewidth=3.0, label='reconstructed signal with SSA')
    # graph_legend.append(l2)
    handles, labels = ax.get_legend_handles_labels()
    handle_as.append(handles)
    labels_as.append(labels)
    plt.title(angle_names[h])
    # leg = plt.legend(handles=handles, labels=labels)

fig.legend(handles=handle_as[0], labels=labels_as[0])
fig.text(0.5, 0.04, 'position', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=20)
plt.show()

#
# pp.savefig(bbox_inches='tight')
# pp.close()


