import threading

import numpy as np

from preprocessing.ssa import SingularSpectrumAnalysis


class PreProcessor(threading.Thread):
    def __init__(self, thread_id, input_buffer, output_buffer, config):
        threading.Thread.__init__(self)
        self.isRun = True
        self.config = config
        self.thread_id = thread_id
        self.lock = threading.Lock()
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.sampling_rate = int(config["sampling_rate"])
        self.train_dir = str(config["train_dir_abs_location"])
        self.number_of_channels = int(config["number_of_channels"])
        self.sampling_time = 1.0 / self.sampling_rate * 1.0
        self.window_size =  int(config["window_size"])

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

        noise_reduced_signal = self.nomalize_signal(noise_signal)
        with open(train_dir + "noise_reduced_signal.csv", 'w') as f:
            np.savetxt(f, noise_reduced_signal, delimiter=',', fmt='%.18e')

        reconstructed_signal = SingularSpectrumAnalysis(noise_reduced_signal,
                                                        self.window_size).execute(number_of_components)
        with open(train_dir + "reconstructed_signal.csv", 'w') as f:
            np.savetxt(f, reconstructed_signal, delimiter=',', fmt='%.18e')
            processed_signal = reconstructed_signal
        with open(train_dir + "feature_vector.csv", 'w') as f:
             np.savetxt(f, np.array(processed_signal), delimiter=',', fmt='%.18e')