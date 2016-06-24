from astrolab_neural_network import AstrolabNeuralNetwork
from astrolab_image_processor import AstrolabImageProcessor
import numpy as np
import sys
import cv2
import os
import traceback

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
            if img.any():
                cv2.imwrite(os.path.join(output_path, file_name), img)

        print("Denoising finished")

    def train(self):
        print("\nTraining")

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
        image_reshaped = np.reshape([image], (-1, image_size * image_size))

        classification = self.neural_network.classify(image_reshaped)
        print("Prediction: " + str(classification[0]))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (400,400), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
