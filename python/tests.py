import os
import cv2
import datetime
import tensorflow as tf
from astrolab_image_processor import AstrolabImageProcessor
from astrolab_neural_network import AstrolabNeuralNetwork
from data import DataWrapper
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

image_size = 56

dataset_folder = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/DATASET/'
train_raw_images = dataset_folder + 'train/images/raw/'
train_processed_images = dataset_folder + 'train/images/processed_' + str(image_size) + '_3/'
train_label_file = dataset_folder + 'train/labels.csv'
test_raw_images = dataset_folder + 'test/images/raw/'
test_processed_images = dataset_folder + 'test/images/processed_' + str(image_size) + '_3/'
test_label_file = dataset_folder + 'test/labels.csv'

# image_processor = AstrolabImageProcessor()
#
# file_name_list_training = sorted(os.listdir(train_raw_images))
#
# for file_name in file_name_list_training:
#     img = cv2.imread(os.path.join(train_raw_images, file_name), cv2.IMREAD_UNCHANGED)
#     img = image_processor.denoise(img, 56)
#     if img != None:
#         cv2.imwrite(os.path.join(train_processed_images, file_name), img)
#
# file_name_list_test = sorted(os.listdir(test_raw_images))
#
# for file_name in file_name_list_test:
#     img = cv2.imread(os.path.join(test_raw_images, file_name), cv2.IMREAD_UNCHANGED)
#     img = image_processor.denoise(img, 56)
#     if img != None:
#         cv2.imwrite(os.path.join(test_processed_images, file_name), img)




# data = input_data.read_data_sets('MNIST_data', one_hot=True)

data = DataWrapper()
data.train.load_labels(train_label_file, 2)
data.train.load_images_names(train_processed_images, image_size)
data.test.load_labels(test_label_file, 2)
data.test.load_images_names(test_processed_images, image_size)


neural_network = AstrolabNeuralNetwork(image_size, 3, 2)

a = datetime.datetime.now()
neural_network.train(data, 50, 100000)
b = datetime.datetime.now()
print("Time taken: " + str(b-a))

# for i in range(0, 50):
#
#     batch = data.test.next_batch(1)
#     print("Class of image is: " + str(batch[1]))
#     neural_network.classify(batch[0])
#
#     image = cv2.imread(os.path.join(test_raw_images, data.test.images_names[i]))
#     resize_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC)
#     cv2.imshow('image', resize_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
