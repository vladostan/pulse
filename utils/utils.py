import numpy as np


def get_label(class_number, number_of_class):
    label = np.zeros(number_of_class, dtype=np.int)
    label[class_number - 1] = 1
    return label