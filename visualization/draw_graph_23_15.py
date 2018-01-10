import json

import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
from scipy import signal
from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('RAW_EMG_SIGNALS.pdf')

index=0
window_size=16
fsamp = 256
tsample = 1 / fsamp
f_low = 128
f_high = 1
order = 2

channels_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]
n_ch = len(channels_names)
df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
df = df[channels_names].dropna(axis=0)

processed_signal = df.copy()
b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))
b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))
Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
b3, a3 = butter(order, Wn, btype='stop')
for i in range(0, n_ch):
    processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))

graph_legend = []
handle_as=[]
labels_as=[]
start = 6000
end = 12000
fsamp = 1
num_ch = len(channels_names)

fig = plt.figure(figsize=(20, 14))
fig.subplots_adjust(hspace=.5)


for h in range(0, num_ch):
    ax = plt.subplot(num_ch,1,h+1)
    axes = plt.gca()
    if(end==0):
        end = processed_signal.ix[:, h].shape[0]-1
    x = np.arange(start, end, 1)
    input_signal = processed_signal.ix[:, h][start * fsamp:end * fsamp]
    l4 = ax.plot(x,input_signal, linewidth=1.5, label='Pre-processed Signal')
    graph_legend.append(l4)

    handles, labels = ax.get_legend_handles_labels()
    handle_as.append(handles)
    labels_as.append(labels)
    plt.title(channels_names[h])
    # leg = plt.legend(handles=handles, labels=labels)

fig.legend(handles=handle_as[0], labels=labels_as[0])
fig.text(0.5, 0.04, 'Sample Count', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'Amplitudes', va='center', rotation='vertical', fontsize=20)
# plt.show()


pp.savefig(bbox_inches='tight')
pp.close()


