from astrolab_neural_network import AstrolabNeuralNetwork
from astrolab_image_processor import AstrolabImageProcessor
from data import DataWrapper
import numpy as np
import sys
import cv2
import os
import traceback
import tensorflow as tf

class AstrolabMain:

    def __init__(self):
        self.neural_network = None
        self.image_processor = AstrolabImageProcessor()
        self.data_wrapper = None

    def start(self):

        print("Welcome to Astrolab")
        while(1):
            try:
                print("\nMenu")
                print("     1 - Denoise")
                print("     2 - Train")
                print("     3 - Classify")
                print("     4 - Exit")
                user_input = input("Choose an option (numeric): ")
                if user_input.isdigit():
                    action = int(user_input)
                    if action == 1:
                        self.denoise()
                    elif action == 2:
                        self.train()
                    elif action == 3:
                        self.classify()
                    elif action == 4:
                        break
                    else:
                        print("Invalid choice")
            except:
                print("\nA error happen:\n" + traceback.format_exc())

    def denoise(self):

        print("\nDenoising")

        image_path = self.read_value(self.is_folder_or_file,"Inform image(s) path: ", "Invalid image(s) path")
        image_size = self.read_value(self.is_integer, "Inform image(s) size: ", "Invalid image(s) size")
        output_path = self.read_value(self.is_folder, "Inform the output folder: ", "Invalid path")

        file_name_list = None

        if self.is_file(image_path):
            image_path_parts = image_path.split('/')
            file_name_list = [image_path_parts[-1]]
            image_path_parts.pop()
            image_path = '/'.join(image_path_parts)
        elif self.is_folder(image_path):
            file_name_list = sorted(os.listdir(image_path))

        for file_name in file_name_list:
            img = cv2.imread(os.path.join(image_path, file_name), cv2.IMREAD_UNCHANGED)
            img = self.image_processor.denoise(img, int(image_size))
            if img.size != 0:
                cv2.imwrite(os.path.join(output_path, file_name), img)

        print("Denoising finished")

    def train(self):

        print("\nTraining")

        train_processed_images = self.read_value(self.is_folder,"Inform train images path: ", "Invalid images path")
        train_label_file = self.read_value(self.is_file, "Inform train labels path: ", "Invalid label file path")
        test_processed_images = self.read_value(self.is_folder, "Inform test images path: ", "Invalid images path")
        test_label_file = self.read_value(self.is_file, "Inform test labels path: ", "Invalid label file path")
        class_size = int(self.read_value(self.is_integer, "Inform the class count: ", "Invalid class count"))

        batch_size = int(self.read_value(self.is_integer, "Inform the batch size: ", "Invalid batch size"))
        iterations = int(self.read_value(self.is_integer, "Inform the iterations number: ", "Invalid iterations size"))

        if not self.neural_network:
            image_size = int(self.read_value(self.is_integer, "Inform the image sizes: ", "Invalid image size"))
            image_channels = int(self.read_value(self.is_integer, "Inform the image channels: ", "Invalid image channels"))
            self.neural_network = AstrolabNeuralNetwork(image_size, image_channels, class_size)
        else:
            image_size = self.neural_network.input_shape
            image_channels = self.neural_network.input_channels

        data = DataWrapper()
        data.train.load_labels(train_label_file, class_size)
        data.train.load_images_names(train_processed_images, image_size)
        data.test.load_labels(test_label_file, class_size)
        data.test.load_images_names(test_processed_images, image_size)

        self.neural_network.train(data, batch_size, iterations)

        print("Training finished")

    def classify(self):
        print("\nClassifying")
        image_path = self.read_value(self.is_file, "Inform image path: ", "Invalid path")

        if not self.neural_network:
            image_size = int(self.read_value(self.is_integer, "Inform the image sizes: ", "Invalid image size"))
            image_channels = int(self.read_value(self.is_integer, "Inform the image channels: ", "Invalid image channels"))
            class_size = int(self.read_value(self.is_integer, "Inform the class count: ", "Invalid class count"))
            self.neural_network = AstrolabNeuralNetwork(image_size, image_channels, class_size)

        image = cv2.imread(image_path)
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_reshaped = np.reshape([image], (-1, self.neural_network.input_dimension))

        classification = self.neural_network.classify(image_reshaped)
        print("\nPrediction: " + str(np.argmax(classification[0])))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (400,400), interpolation = cv2.INTER_CUBIC)

        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", image)


    def read_value(self, valid_function, message, error_message):
        value = input(message)
        while not valid_function(value):
            print(error_message)
            value = input(message)
        return value

    def is_file(self, path):
        return os.path.isfile(path)

    def is_folder(self, path):
        return os.path.isdir(path)

    def is_folder_or_file(self, path):
        return os.path.isfile(path) or os.path.isdir(path)

    def is_integer(self, text):
        return text.isdigit()
