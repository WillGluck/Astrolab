import os
import cv2
import csv
import numpy as np

class DataWrapper:

    def __init__(self):
        self.train = Data()
        self.test = Data()


class Data:

    def __init__(self):
        self.images_names = []
        self.labels = []
        self.current_index = 0
        self.images_path = ""
        self.image_dimension = 0

    def is_batch_size_valid(self, batch_size):
        return self.current_index + batch_size > len(self.images_names)

    def create_joined_list(self, batch_size, source_list, update_index = False):
        first_part = source_list[self.current_index : len(source_list)]
        last_part = source_list[0 : batch_size - len(first_part)]
        if update_index:
            self.current_index = batch_size - len(first_part)
        return first_part + last_part

    def next_batch(self, batch_size):

        images_batch = None
        labels_batch = None

        #TODO verificar batch maior que o tamanho da lista.
        if self.is_batch_size_valid(batch_size):

            images_batch = self.create_joined_list(batch_size, self.images_names)
            labels_batch = self.create_joined_list(batch_size, self.labels, True)

        else:

            final_index = self.current_index + batch_size
            images_batch = self.images_names[self.current_index : final_index]
            labels_batch = self.labels[self.current_index : final_index]

            self.current_index = final_index

        return self.load_images_from_names(images_batch), labels_batch

    def load_images_from_names(self, images_names):
        images = []
        for name in images_names:
            images.append(cv2.imread(os.path.join(self.images_path, name), 0))
        return np.reshape(images, (-1, self.image_dimension))


    def load_labels(self, path):
        #'/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/'
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:

                content = None
                content = list(row[i] for i in [0, 1])

                value = [0, 0, 0]
                value[int(content[1])] = 1
                self.labels.append(value)
        print(self.labels)

    def load_images_names(self, path, image_dimension):
        #'/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/images/'
        self.images_path = path
        self.image_dimension = image_dimension
        self.images_names = sorted(os.listdir(path))
        print(self.images_names)
