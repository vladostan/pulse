from __future__ import print_function
import Queue
import json
import socket
import sys
import threading
import tensorflow as tf
import pandas as pd
from scipy.signal import butter, filtfilt

import librosa
import numpy as np

from ssa import SingularSpectrumAnalysis


class PreProcessor(threading.Thread):
    def __init__(self, thread_id, input_buffer, output_buffer, config):
        threading.Thread.__init__(self)
        self.isRun = True
        self.config = config
        self.thread_id = thread_id
        self.lock = threading.Lock()
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.window_size = int(config["window_size"])
        self.sampling_rate = int(config["sampling_rate"])
        self.low_frequency = int(config["low_frequency"])
        self.high_frequency = int(config["high_frequency"])
        self.order = int(config["order"])
        self.train_dir = str(config["train_dir_abs_location"])
        self.number_of_channels = int(config["number_of_channels"])
        self.sampling_time = 1.0 / self.sampling_rate * 1.0

    def run(self):
        self.processor(self.thread_id)

    def nomalize_signal(self, input_signal):
        mean = np.mean(input_signal, axis=0)
        input_signal -= mean
        return input_signal / np.std(input_signal, axis=0)

    def processor(self, thread_id, activity_type=None, number_of_components=None):
        noise_signal = self.input_buffer[thread_id]
        noise_signal = noise_signal[~np.isnan(noise_signal)]
        if activity_type is not None:
            train_dir = self.train_dir + "/" + str(thread_id) + "_" + activity_type + "_"
        else:
            train_dir = self.train_dir + "/" + str(thread_id) + "_"

        with open(train_dir + "noise_signal.csv", 'w') as f:
            np.savetxt(f, noise_signal, delimiter=',', fmt='%.18e')
        # noise_reduced_signal = self.apply_noise_reducer_filer(noise_signal)

        noise_reduced_signal = self.nomalize_signal(noise_signal)
        with open(train_dir + "noise_reduced_signal.csv", 'w') as f:
            np.savetxt(f, noise_reduced_signal, delimiter=',', fmt='%.18e')

        reconstructed_signal = SingularSpectrumAnalysis(noise_reduced_signal,
                                                        self.window_size).execute(number_of_components)
        with open(train_dir + "reconstructed_signal.csv", 'w') as f:
            np.savetxt(f, reconstructed_signal, delimiter=',', fmt='%.18e')
        # todo uncomment when you running the main process
        processed_signal = []
        position = 0
        # for i in range(0, int((noise_reduced_signal.shape[0]) - int(self.config['window_size']) - 1)):
        #     clip = Clip(self.config, buffer=np.array(noise_reduced_signal[position:position + int(self.config['window_size'])].tolist()))
        #     processed_signal.append(clip.get_feature_vector())
        #     position += 1
        processed_signal = reconstructed_signal
        with open(train_dir + "feature_vector.csv", 'w') as f:
             np.savetxt(f, np.array(processed_signal), delimiter=',', fmt='%.18e')
            # self.lock.acquire()
            # self.output_buffer[thread_id] = reconstructed_signal
            # self.lock.release()

    def apply_noise_reducer_filer(self, data):
        data = np.array(data, dtype=float)
        b, a = butter(self.order, (self.order * self.low_frequency * 1.0)
                      / self.sampling_rate * 1.0, btype='low')
        # for i in range(0, self.number_of_channels):
        data = np.transpose(filtfilt(b, a, data))

        b1, a1 = butter(self.order, (self.order * self.high_frequency * 1.0) /
                        self.sampling_rate * 1.0, btype='high')
        # for i in range(0, self.number_of_channels):
        data = np.transpose(filtfilt(b1, a1, data))

        Wn = (np.array([58.0, 62.0]) / 500 * self.order).tolist()
        b3, a3 = butter(self.order, Wn, btype='stop')
        for i in range(0, self.number_of_channels):
            data = np.transpose(filtfilt(b3, a3, data))

        Wn = [0.05008452488,0.152839]
        b3, a3 = butter(self.order, Wn, btype='stop')
        for i in range(0, self.number_of_channels):
            data = np.transpose(filtfilt(b3, a3, data))

        return data
