import json
import threading

import numpy as np

import init_buffer as buf
from preprocessing import PreProcessor
# import draw_sample_plot_and_save, create_sample_from_image
# import get_label
from utils.dataset_writer_utils import draw_sample_plot_and_save, create_sample_from_image
from utils.utils import get_label


class NoiseReducer(threading.Thread):
    def __init__(self, thread_id, server, writer, config):
        threading.Thread.__init__(self)
        self.config = config
        self.window_size = int(config["window_size"])
        self.verbose = eval(config["verbose"])
        self.input_data = buf.ring_buffers
        self.number_of_threads = int(config["number_of_channels"])
        self.feature_vector_size = int(config["feature_vector_size"])
        self.train_dir = str(config["train_dir_abs_location"])
        self.counter=0
        # self.train_dir = str(config["train_dir"])
        self.lock = threading.Lock()
        self.input_buffer = np.zeros([self.number_of_threads, self.window_size])
        self.thread_id = thread_id
        self.output_buffer = []
        self.is_processing = False
        self.server = server
        self.writer = writer
        self.number_of_class = int(config["processing"]["train"]["number_of_class"])
        self.ip = str(config["ip"])
        self.port = int(config["port"]) + 5  # adding five offset to secondary udp server
        self.overlap_size = int(config["overlap_size"])

    def construct_input_buffer(self):
        for j in range(0, len(self.input_data)):
            try:
                self.input_buffer[j] = self.input_data[j].pop_window(self.window_size, self.overlap_size)
            except:
                print ("Still input buffer is empty... creating random data set...")
                self.input_buffer[j] = [i for i in range(0, self.window_size)]
                pass
        self.input_buffer = np.array(self.input_buffer)

    def run(self):
        if self.verbose:
            print("Starting " + str(self.thread_id))
        self.lock.acquire()
        self.is_processing = True
        self.construct_input_buffer()
        self.process_signal()
        if self.verbose:
            print (self.output_buffer)
        self.is_processing = False
        self.lock.release()
        if self.verbose:
            print("Existing " + str(self.thread_id))

    def process_signal(self):
        self.counter += 1
        self.output_buffer = np.zeros([self.input_buffer.shape[0], self.feature_vector_size])
        threads = []
        thread_list = [i for i in range(0, self.number_of_threads)]
        for thread_id in thread_list:
            thread = PreProcessor(thread_id, self.input_buffer, self.output_buffer, config=self.config)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        # with open(self.train_dir + "/feature_vectors.csv", 'a') as f:
        #         np.savetxt(f, self.output_buffer, delimiter=',', fmt='%.18e')

        clip_label = get_label(1, self.number_of_class)
        clip_filename = draw_sample_plot_and_save(self.output_buffer.flatten(), "/channel", self.thread_id, self.config)
        sample = create_sample_from_image(clip_filename, clip_label, self.config)
        # sample = create_sample_from_data(self.output_buffer.flatten(), class_label)
        self.writer.write(sample.SerializeToString())
        self.send_noise_data(json.dumps(self.input_buffer.tolist()))
        self.send_preprocessed_data(json.dumps(self.output_buffer.tolist()))
        # return self.output_buffer

    def send_preprocessed_data(self, data):
        self.server.sendto(data, (self.ip, self.port))

    def send_noise_data(self, data):
        self.server.sendto(data, (self.ip, self.port+1))

# project_file_path = "/home/runge/openbci/OpenBCI_Python"
# config_file = "/home/runge/openbci/OpenBCI_Python/config/config.json"
#
# with open(config_file) as config:
#             plugin_config = json.load(config)
#             buffer_size = int(plugin_config["buffer_size"])
#             number_of_channels = int(plugin_config["number_of_channels"])
#             buffer_capacity = int(plugin_config["buffer_capacity"])
#             tfrecords_filename = project_file_path + str(plugin_config["model"]["tfrecords_filename"])
#             lock = threading.Lock()
#             server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#             ring_buffers = [RingBuffer(buffer_size * 4) for i in range(0, number_of_channels)]
#             for k in range(0, number_of_channels):
#                 for p in range(0, buffer_size*buffer_capacity):
#                     ring_buffers[k].append(random.randint(1,100))
#             writer = tf.python_io.TFRecordWriter(t ,mfrecords_filename)
#             noisereducer_thread =  NoiseReducer("main thread",ring_buffers,server,lock, writer, plugin_config)
#             i = 0
#             while i<100:
#                 if not noisereducer_thread.is_processing:
#                     print ("------current process-----")
#                     noisereducer_thread = NoiseReducer("main thread", ring_buffers,server,lock, writer, plugin_config)
#                     noisereducer_thread.start()
#                     noisereducer_thread.join()
#                     i+=1
#             writer.close()



