import sys

sys.path.insert(0, 'core/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'general/')

import os
import cv2
import datetime
import tensorflow as tf
from astrolab_image_processor import AstrolabImageProcessor
from astrolab_neural_network import AstrolabNeuralNetwork
from data import DataWrapper
from matplotlib import pyplot as plt
import math
import numpy as np

def truncate(number, resolution):
    return int(np.round(number/resolution))*resolution

from tensorflow.examples.tutorials.mnist import input_data

image_size = 56

dataset_folder = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/'
train_raw_images = dataset_folder + 'train/images/raw/'
train_processed_images = dataset_folder + 'train/images/processed_' + str(image_size) + '_3/'
train_label_file = dataset_folder + 'train/labels/' + str(image_size) + '/labels.csv'
test_raw_images = dataset_folder + 'test/images/raw/'
test_processed_images = dataset_folder + 'test/images/processed_' + str(image_size) + '_3/'
test_label_file = dataset_folder + 'test/labels/' + str(image_size) + '/labels.csv'

image_processor = AstrolabImageProcessor()

file_name_list_training = sorted(os.listdir(train_raw_images))

# for file_name in file_name_list_training:
#     img = cv2.imread(os.path.join(train_raw_images, file_name), cv2.IMREAD_UNCHANGED)
#     img = image_processor.denoise(img, image_size)
#     if img.size != 0:
#         cv2.imwrite(os.path.join(train_processed_images, file_name), img)
#
# file_name_list_test = sorted(os.listdir(test_raw_images))
#
# for file_name in file_name_list_test:
#     img = cv2.imread(os.path.join(test_raw_images, file_name), cv2.IMREAD_UNCHANGED)
#     img = image_processor.denoise(img, image_size)
#     if img.size != 0:
#         cv2.imwrite(os.path.join(test_processed_images, file_name), img)
#

img = cv2.imread("/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/1581.jpg", cv2.IMREAD_UNCHANGED)
img = image_processor.denoise(img, image_size)

# data = input_data.read_data_sets('MNIST_data', one_hot=True)

# data = DataWrapper()
# data.train.load_labels(train_label_file, 2)
# data.train.load_images_names(train_processed_images, image_size)
# data.test.load_labels(test_label_file, 2)
# data.test.load_images_names(test_processed_images, image_size)

# pre_type_train = 0
# late_type_train = 0
#
# for i in range (0, len(data.train.images_names)):
#     batch = data.train.next_batch(1)
#     if batch[1][0][1] == 1:
#         late_type_train += 1
#     else:
#         pre_type_train += 1
#
# print("Pre type: " + str(pre_type_train))
# print("Late type: " + str(late_type_train))
#
# pre_type_test = 0
# late_type_test = 0
#
# for i in range (0, len(data.test.images_names)):
#     batch = data.test.next_batch(1)
#     if batch[1][0][1] == 1:
#         late_type_test += 1
#     else:
#         pre_type_test += 1
#
# print("Pre type: " + str(pre_type_test))
# print("Late type: " + str(late_type_test))

# neural_network = AstrolabNeuralNetwork(image_size, 3, 2)
#
# a = datetime.datetime.now()
# neural_network.train(data, 50, 100000)
# b = datetime.datetime.now()
# print("Time taken: " + str(b-a))

# error_pre = 0
# error_late = 0
#
# for i in range(0, 10):
#
#     batch = data.test.next_batch(1)
#     real_class = ''
#     class_index = 0
#     if batch[1][0][1] == 1:
#         real_class = 'Late type'
#         class_index = 1
#     else:
#         real_class = 'Young type'
#
#     # print("Class of image is " + real_class)
#     prediction = neural_network.classify(batch[0])
#     # print("Prediction is: \n" + str(truncate(prediction[0][0] * 100, 0.01)) + "% young type\n" + str(truncate(prediction[0][1] * 100, 0.01)) + "% late type\n")
#
#     if prediction[0][0] > prediction[0][1] and class_index == 1:
#         error_late += 1
#     elif prediction[0][1] > prediction[0][0] and class_index == 0:
#         error_pre += 1

    # image = cv2.imread(os.path.join(test_raw_images, data.test.images_names[i]))
    # resize_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC)
    # cv2.imshow('image', resize_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# print("Pre galaxies error: " + str(error_pre))
# print("Late galaxies error: " + str(error_late))
