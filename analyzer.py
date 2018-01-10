from __future__ import print_function

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.decomposition import FastICA, PCA

from lib.dtw import dtw, fastdtw

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import manhattan_distances
from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis


class SignalAnalyzer():
    def __init__(self, activity_type, project_path, dataset_location):
        self.raw_data = pd.read_csv(dataset_location)
        self.config_file = project_path + "/config/config.json"
        self.raw_data = self.raw_data.ix[:, 0:5].dropna()
        self.raw_channel_data = self.raw_data.ix[:, 0:5]
        self.raw_kiynect_angle_data = self.raw_data.ix[:, 0:3]
        self.channel_length = self.raw_channel_data.shape[1]
        self.kinect_angle_length = 3
        self.angle_names = ["wrist", "elbow", "shoulder"]
        self.signal_types = ["noise_signal", "noise_reduced_signal", "feature_vector"]
        self.raw_channel_data_set = []
        self.output_buffer = []
        self.activity_type = activity_type
        self.project_path = project_path
        self.dataset_location = dataset_location
        self.channels_names = ["component1", "component2", "component3", "component4", "component5"]
        with open(self.config_file) as config:
            self.config = json.load(config)
            self.config["train_dir_abs_location"] = self.project_path + "/build/dataset/train"

    def nomalize_signal(self, input_signal):
        mean = np.mean(input_signal, axis=0)
        input_signal -= mean
        return input_signal / np.std(input_signal, axis=0)

    def data_reprocessing(self, number_of_pin_componenets, activity_type):
        for i in range(0, self.channel_length):
            self.raw_channel_data_set.append(self.nomalize_signal(self.raw_channel_data.ix[:, i]))
        for i in range(0, self.channel_length):
            preprocessor = PreProcessor(i, self.raw_channel_data_set, self.output_buffer, self.config)
            preprocessor.processor(i, activity_type=activity_type, number_of_components=number_of_pin_componenets)

    def signal_combine(self, activity_type):
        for i in range(0, len(self.signal_types)):
            signal_type = self.signal_types[i]
            noise_signals = []
            for i in range(0, self.channel_length):
                processed_signal = pd.read_csv(str(self.config["train_dir_abs_location"]) + "/" + str(i) + "_" +
                                               activity_type + "_" + signal_type + ".csv")
                noise_signals.append(np.array(processed_signal.ix[:, 0]).astype(np.float64))
            with open(str(self.config[
                              "train_dir_abs_location"]) + "/result/" + activity_type + "_" + signal_type + "s" + ".csv",
                      'w') as f:
                np.savetxt(f, np.transpose(np.array(noise_signals)), delimiter=',', fmt='%.18e')

    def inti_test_data(self):
        np.random.seed(0)
        n_samples = 3000
        time = np.linspace(0, 10, n_samples)

        s1 = np.sin(2 * time)
        s2 = np.sign(np.sin(3 * time))
        s3 = signal.sawtooth(2 * np.pi * time)
        s4 = np.sign(np.sin(3.2 * time))
        s5 = signal.sawtooth(2.9 * np.pi * time)

        noise_signals = []
        noise_signals.append(s1)
        noise_signals.append(s2)
        noise_signals.append(s3)
        noise_signals.append(s4)
        noise_signals.append(s5)

        with open(str(self.config[ "train_dir_abs_location"]) + "/result/" + "initial_tests" + ".csv",'w') as f:
                np.savetxt(f, np.transpose(np.array(noise_signals)), delimiter=',', fmt='%.18e')

    def plot_signals(self, is_save=False, start=0, end=0, fsamp=1, is_raw=False, is_compare=False):
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        matplotlib.rc('axes', titlesize=15)
        matplotlib.rc('legend', fontsize=15)
        if is_raw:
            raw_channels_data = pd.read_csv(self.dataset_location).ix[:, 2:7].dropna()
        else:
            raw_channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
        noise_reducer_signal_data = pd.read_csv(self.config["train_dir_abs_location"]
                                        + "/result/"+self.activity_type+"_noise_reduced_signals.csv").dropna()
        self.save_channels = PdfPages('channels_'+self.activity_type+'_reconstructed.pdf')
        graph_legend = []
        handle_as = []
        labels_as = []
        num_ch = len(self.channels_names)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=.5)

        index = 1
        num_types = 1

        if is_compare:
            num_types = 2
        for h in range(0, num_ch):
            # preprocessor = PreProcessor(h, None, None, self.config)
            ax = plt.subplot(num_ch*num_types, num_types, index)
            if (end == 0):
                end = raw_channels_data.ix[:, h].shape[0] - 1
            x = np.arange(start, end, 1)
            input_signal = raw_channels_data.ix[:, h][start * fsamp:end * fsamp]
            noise_reduced_signal = noise_reducer_signal_data.ix[:, h][start * fsamp:end * fsamp]

            l1 = ax.plot(noise_reduced_signal, linewidth=1.0, label="raw signal")
            graph_legend.append(l1)

            index+=1
            if is_compare:
                ax = plt.subplot(num_ch * num_types, num_types, index)
                l2 = ax.plot(input_signal, linewidth=1.0, label="svd signal")
                graph_legend.append(l2)
                index += 1

            # with open("input.csv", 'w') as f:
            #     np.savetxt(f, input_signal, delimiter=',', fmt='%.18e')

            # noise_reducer_signal = preprocessor.apply_noise_reducer_filer(input_signal)
            # l2 = ax.plot(x, noise_reducer_signal, linewidth=3.0, label="noise_reducer_signal")
            # graph_legend.append(l2)

            # normalize_signal = preprocessor.nomalize_signal(noise_reducer_signal)
            # l3 = ax.plot(x, normalize_signal, linewidth=1.0, label="normalize_signal")
            # graph_legend.append(l3)

            # reconstructed_signal = SingularSpectrumAnalysis(noise_reducer_signal, self.config["window_size"], False).execute(1)
            # l4 = ax.plot(x,reconstructed_signal, linewidth=1.0, label='reconstructed signal with SSA')
            # graph_legend.append(l4)

            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[h])
            # leg = plt.legend(handles=handles, labels=labels)

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
        fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
        fig.tight_layout()
        if is_save:
            self.save_channels.savefig(bbox_inches='tight')
            self.save_channels.close()
        else:
            plt.show()


    def apply_dwt_for_each_channals(self, nomalized_signal, start, end, pattern_start_at, pattern_end_at, is_apply_dwt, channel_number_to_plot):
        if(is_apply_dwt):
            for i in range(0, self.channel_length):
                channel_number = i
                pattern = np.array(nomalized_signal.ix[:, channel_number][pattern_start_at:pattern_end_at])
                result = []
                possion = []
                final_result = []
                size = pattern_end_at - pattern_start_at
                counter = start
                for i in range(0, int(np.floor((end-start)/5))):
                    y = np.array(nomalized_signal.ix[:, channel_number][counter:counter + size])
                    possion.append(counter)
                    #counter += int(np.math.ceil(size/2))
                    counter += 5
                    # dist, cost, acc, path = dtw(pattern, y, manhattan_distances)
                    dist, cost, acc, path = fastdtw(pattern, y, 'euclidean')
                    print (dist)
                    result.append(dist)
                final_result.append(result)
                final_result.append(possion)
                with open(self.config["train_dir_abs_location"] + "/result/"+self.activity_type+"_dwt_result_"
                                  +str(channel_number)+".csv", 'w') as f:
                    np.savetxt(f, np.transpose(np.array(final_result)), delimiter=',', fmt='%.18e')
        else:
            dwt_result = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_dwt_result_"+str(channel_number_to_plot)
                                     +".csv").dropna()
            return dwt_result.ix[:,0], dwt_result.ix[:,1]

    def apply_dwt(self, nomalized_signal, start, end, pattern_start_at, pattern_end_at, is_apply_dwt, channel_number=1):
        if(is_apply_dwt):
            pattern = np.array(nomalized_signal.ix[:, channel_number][pattern_start_at:pattern_end_at])
            result = []
            possion = []
            final_result = []
            size = pattern_end_at - pattern_start_at
            counter = start
            for i in range(0, int(np.floor((end-start)/5))):
            # for i in range(0, 3):
            #     y = np.array(nomalized_signal.ix[:, channel_number][counter:counter + size]).tolist()
                y = np.array(nomalized_signal.ix[:, channel_number][counter:counter + size])
                possion.append(counter)
                counter += 5
                # dist, cost, acc, path = dtw(pattern, y, manhattan_distances)
                dist, cost, acc, path = fastdtw(pattern, y, 'euclidean')
                print (dist)
                result.append(dist)
            final_result.append(result)
            final_result.append(possion)

            with open(self.config["train_dir_abs_location"] + "/result/"+self.activity_type+"_dwt_result.csv", 'w') as f:
                np.savetxt(f, np.transpose(np.array(final_result)), delimiter=',', fmt='%.18e')
            return result, possion
        else:
            dwt_result = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_dwt_result.csv").dropna()
            return dwt_result.ix[:,0], dwt_result.ix[:,1]

    def plot_processed_singals_by_ssa(self, start=0, end=0, fsamp=1, is_raw=False):
        channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
        graph_legend = []
        handle_as = []
        labels_as = []

        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=.5)
        if end==0:
            end= channels_data.ix[:, 0].shape[0] - 1

        x = np.arange(start, end, 1)
        for i in range(0, 5):
            ax = plt.subplot(510 + i + 1)
            l1 = ax.plot(channels_data.ix[:, i][start:end], linewidth=1.0, label="Processed signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[i])

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'Position', ha='center', fontsize=10)
        fig.text(0.04, 0.5, 'Signal Amplitude', va='center', rotation='vertical', fontsize=10)
        plt.show()


    def plot_initial_signals(self, start=0, end=0, signal_type='noise_signals',fsamp=1, is_raw=False, with_ssa=False):
        channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_"+signal_type+".csv").dropna()
        noise_removed_data = pd.read_csv(self.config["train_dir_abs_location"]
                                    + "/result/" + self.activity_type + "_" + "feature_vectors" + ".csv").dropna()


        graph_legend = []
        handle_as = []
        labels_as = []

        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=.5)
        if end==0:
            end = channels_data.ix[:, 0].shape[0] - 1
        for i in range(0, 5):
            ax = plt.subplot(510 + i + 1)
            l1 = ax.plot(channels_data.ix[:, i][start:end], linewidth=1.0, label="Initial Signal")
            if with_ssa:
                l1 = ax.plot(noise_removed_data.ix[:, i][start:end], linewidth=1.0, label="Processed Signal with SSA")
            graph_legend.append(l1)
            handles, labels = ax.get_legend_handles_labels()
            handle_as.append(handles)
            labels_as.append(labels)
            plt.title(self.channels_names[i])

        fig.legend(handles=handle_as[0], labels=labels_as[0])
        fig.text(0.5, 0.04, 'Position', ha='center', fontsize=20)
        fig.text(0.04, 0.5, 'Signal Amplitude', va='center', rotation='vertical', fontsize=20)
        plt.show()
        self.correlation_matrix(channels_data)

    def correlation_matrix(self, df):
        from matplotlib import pyplot as plt
        from matplotlib import cm as cm

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cmap = cm.get_cmap('jet', 30)
        cax = ax1.imshow(df.corr(), cmap=cmap)
        #ax1.grid(True)
        #plt.title('Feature Correlation')
        labels = self.channels_names
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_yticklabels(labels, fontsize=10)
        fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
        plt.show()

    def select_the_best_component(self, start=0, end=0, fsamp=1, is_raw=False, pattern_start_at=0,
                                  pattern_end_at=200, is_apply_dwt=False, channel_number_to_plot=1,
                                  theshold_level=0.5, is_plot=False):
        channels_data = pd.read_csv(self.config["train_dir_abs_location"]
                                            + "/result/"+self.activity_type+"_feature_vectors.csv").dropna()
        nomalized_signal = self.nomalize_signal(channels_data)
        if end==0:
            end = nomalized_signal.shape[0] - 1

        if(is_apply_dwt):
            self.apply_dwt_for_each_channals(nomalized_signal, start, end, pattern_start_at, pattern_end_at,
                                             is_apply_dwt, channel_number_to_plot)

        is_apply_dwt=False
        from scipy.stats import moment
        third_momentoms=[]
        indices_of_channels = []
        for k in range(0, self.channel_length):
            channel_number_to_plot = k
            distance, possion = self.apply_dwt_for_each_channals(nomalized_signal, start, end, pattern_start_at, pattern_end_at,
                                               is_apply_dwt, channel_number_to_plot)
            maxtab, mintab = self.lowest_point_detect(distance, theshold_level)

            if len(mintab)==0:
                print ("No patterns were detected...")
            else:
                detection_points = maxtab[:, 0]
                difference = [abs(j - i) for i, j in zip(detection_points, detection_points[1:])]
                third_momentom = moment(difference, moment=3)

                third_momentoms.append(third_momentom)
                indices = possion[np.array(maxtab[:, 0], dtype=int)]
                indices_of_channels.append(indices)
                if is_plot:
                    fig = plt.figure(figsize=(15, 10))
                    fig.subplots_adjust(hspace=.5)
                    handle_as1 = []
                    labels_as1 = []
                    ax = plt.subplot(111)
                    l1 = ax.plot(distance, linewidth=1.0, label="Processed signal with MDTW")
                    ax.scatter(np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], color='red')
                    handles, labels = ax.get_legend_handles_labels()
                    handle_as1.append(handles)
                    labels_as1.append(labels)
                    fig.legend(handles=handle_as1[0], labels=labels_as1[0])
                    fig.text(0.5, 0.04, 'Position', ha='center', fontsize=20)
                    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=20)

                    graph_legend = []
                    handle_as = []
                    labels_as = []

                    fig = plt.figure(figsize=(15, 10))
                    fig.subplots_adjust(hspace=.5)
                    x = np.arange(start, end, 1)
                    for i in range(0, 5):
                        ax = plt.subplot(510 + i + 1)
                        l1 = ax.plot(x, self.nomalize_signal(channels_data.ix[:, i][start:end]), linewidth=1.0,
                                     label="Processed signal with SSA")
                        graph_legend.append(l1)
                        handles, labels = ax.get_legend_handles_labels()
                        handle_as.append(handles)
                        labels_as.append(labels)
                        plt.title(self.channels_names[i])
                        for i in indices:
                            plt.plot([i, i], [2,1], '-r')

                    fig.legend(handles=handle_as[0], labels=labels_as[0])
                    fig.text(0.5, 0.04, 'Position', ha='center', fontsize=20)
                    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=20)

        third_momentoms = np.array(third_momentoms)
        selected_channel = np.where(third_momentoms==third_momentoms.min())
        plt.show()
        return indices_of_channels[selected_channel[0][0]], third_momentoms, selected_channel

    def lowest_point_detect(self, v, delta, x=None):
        maxtab = []
        mintab = []
        if x is None:
            x = np.arange(len(v))
        v = np.asarray(v)
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx - delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        return np.array(maxtab), np.array(mintab)

    def execute(self, number_of_pin_componenets, activity_type, is_init=False):
        start = 0
        end = 0
        if is_init:
            self.data_reprocessing(number_of_pin_componenets, activity_type)
            self.signal_combine(activity_type)
        self.plot_processed_singals_by_ssa(start, end, is_raw=False)



project_path = "/home/runge/project/pulse"

#label = "g1"
label = "g2"
#label = "v1"
#label = "v2"

technique_type="pca"
#technique_type="jade"
#technique_type="shibbs"
#technique_type="kica"
#technique_type="fica"
#technique_type="mkica"



technique_type_and_label = label + technique_type
dataset_location = project_path + "/build/dataset/"+technique_type_and_label+".csv"
number_of_pin_componenets=1

signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location)
signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
#signal_analyzer.plot_initial_signals(start=0, end=300, with_ssa=False)
peak_points, selector, selected_channel = signal_analyzer.select_the_best_component(pattern_start_at=0, pattern_end_at=1000, start=0, end=0,
                                          is_apply_dwt=True, channel_number_to_plot=0, theshold_level=0.1,
                                          is_plot=True)
print(peak_points)
print(selector)
print("Selected channel: "+ str(selected_channel[0]+1))




