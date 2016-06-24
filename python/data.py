import os
import cv2
import csv
import numpy as np
from util import Util
import tensorflow as tf

class DataWrapper:

    def __init__(self, saved_variables_path):
        self.train = Data()
        self.test = Data()

class Data:

    def __init__(self):
        self.images_names = []
        self.labels = []
        self.current_index = 0
        self.images_path = ""
        self.image_dimension = 0

    def is_batch_size_invalid(self, batch_size):
        return self.current_index + batch_size > len(self.images_names)

    def create_joined_list(self, batch_size, source_list, update_index = False):
        first_part = source_list[self.current_index :]
        last_part = source_list[0 : batch_size - len(first_part)]
        if update_index:
            self.current_index = batch_size - len(first_part)
        return first_part + last_part

    def next_batch(self, batch_size):

        images_batch = None
        labels_batch = None

        #TODO verificar batch maior que o tamanho da lista.
        if self.is_batch_size_invalid(batch_size):
            images_batch = self.create_joined_list(batch_size, self.images_names)
            labels_batch = self.create_joined_list(batch_size, self.labels, True)

        else:

            final_index = self.current_index + batch_size
            images_batch = self.images_names[self.current_index : final_index]
            labels_batch = self.labels[self.current_index : final_index]

            self.current_index = final_index

        # print(images_batch)
        # print(labels_batch)
        return self.load_images_from_names(images_batch), labels_batch


    def load_images_from_names(self, images_names):
        # images = []
        # for name in images_names:
        #     images.append(cv2.imread(os.path.join(self.images_path, name), 0))

        images = []
        for name in images_names:
            image = cv2.imread(os.path.join(self.images_path, name))
            images.append(cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))

        return np.reshape(images, (-1, self.image_dimension))


    def load_labels(self, path, labels_size):
        #'/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/'
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:

                content = None
                content = list(row[i] for i in [0])

                value = np.zeros(labels_size)
                value[int(content[0])] = 1
                self.labels.append(value)
        print("labels count: " + str(len(self.labels)))

    def load_images_names(self, path, image_dimension):
        #'/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/images/'
        self.images_path = path
        self.image_dimension = image_dimension * image_dimension
        self.images_names = Util.sort_nicely(os.listdir(path))
        print("images count: " + str(len(self.images_names)))
