import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)
import Image
import numpy as np
import tensorflow as tf
import librosa.display

from data_types_utils import _int64_feature, _bytes_feature


def read_tf_recode(config):
    reconstructed_clips = []
    record_iterator = tf.python_io.tf_record_iterator(path=config.tfrecords_filename)
    for string_record in record_iterator:
        raw_clip = tf.train.Example()
        raw_clip.ParseFromString(string_record)
        height = int(raw_clip.features.feature['clip_height'].int64_list.value[0])
        width = int(raw_clip.features.feature['clip_width'].int64_list.value[0])
        img_string = (raw_clip.features.feature['clip_raw'].bytes_list.value[0])
        label = (raw_clip.features.feature['clip_label_raw'].bytes_list.value[0])
        img_1d = np.fromstring(img_string, dtype=np.float64)
        label = np.fromstring(img_string, dtype=np.uint64)
        reconstructed_clip = img_1d.reshape((height, width, -1))
        reconstructed_clip_label = label.reshape((1, config.number_of_class, -1))
        reconstructed_clips.append((reconstructed_clip, reconstructed_clip_label))
    return reconstructed_clips

def create_sample_from_data(clip, clip_label):
    feature_vector = clip.tostring()
    clip_label = clip_label.tostring()
    return tf.train.Example(features=tf.train.Features(feature={
        'clip_height': _int64_feature(1),
        'clip_width': _int64_feature(clip.shape[0]),
        'clip_raw': _bytes_feature(feature_vector),
        'clip_label_raw': _bytes_feature(clip_label)}))


def create_sample_from_image(clip_filename, clip_label, config):
    image_width = int(config["processing"]["train"]["generated_image_width"])
    image_height = int(config["processing"]["train"]["generated_image_height"])
    image = Image.open(clip_filename)
    image = image.resize((image_width, image_height),Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.flatten()
    # np.reshape(image, (-1, image.shape[1] * image.shape[0]))
    feature_vector = image.tostring()
    clip_label = clip_label.tostring()
    clip_raw = tf.train.Example(features=tf.train.Features(feature={
        'clip_height': _int64_feature(image.shape[0]),
        'clip_width': _int64_feature(image.shape[0]),
        'clip_raw': _bytes_feature(feature_vector),
        'clip_label_raw': _bytes_feature(clip_label)}))
    return clip_raw


def draw_sample_plot_and_save(clip, clip_type, index, config):
    result = []
    image_width=int(config["processing"]["train"]["generated_image_width"])
    image_height=int(config["processing"]["train"]["generated_image_height"])
    figure = plt.figure(figsize=(
        np.ceil(image_width + image_width * 0.2),
        np.ceil(image_height + image_height * 0.2)), dpi=1)
    axis = figure.add_subplot(111)
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',
                    labelleft='off',
                    labeltop='off',
                    labelright='off', labelbottom='off')
    result.append(clip)
    result = np.array(result)
    librosa.display.specshow(result, sr=int(config["sampling_rate"]), x_axis='time', y_axis='mel', cmap='RdBu_r')
    extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    clip_filename = "%s%s%s%s" % (config["train_dir_abs_location"], clip_type, str(index), "_.jpg")
    plt.savefig(clip_filename, format='jpg', bbox_inches=extent, pad_inches=0)
    plt.close(figure)
    return clip_filename

def read_and_decode(filename_queue, config):
    image_width = int(config["processing"]["train"]["generated_image_width"])
    image_height = int(config["processing"]["train"]["generated_image_height"])
    number_of_channels = int(config["number_of_channels"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'clip_height': tf.FixedLenFeature([], tf.int64),
            'clip_width': tf.FixedLenFeature([], tf.int64),
            'clip_raw': tf.FixedLenFeature([], tf.string),
            'clip_label_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['clip_raw'], tf.int8)
    label = tf.decode_raw(features['clip_label_raw'], tf.int64)
    image = tf.reshape(image, [1, image_width*image_height*3])
    image = tf.cast(image, tf.float32)
    label = tf.reshape(label, [1, int(config["processing"]["train"]["number_of_class"])])
    return image, label


# def inputs(config):
#     with tf.name_scope('input'):
#         filename_queue = tf.train.string_input_producer([config.tfrecords_filename],
#                                                         num_epochs=config.num_epochs)
#         image, label = read_and_decode(filename_queue, config)
#         images, sparse_labels = tf.train.shuffle_batch(
#             [image, label], batch_size=config.batch_size, num_threads=config.batch_process_threads_num,
#             capacity=1000 + 3 * config.batch_size,
#             min_after_dequeue=100)
#         return images, sparse_labels


