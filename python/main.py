import os
import cv2
import tensorflow
from astrolab_image_processor import AstrolabImageProcessor
from astrolab_neural_network import AstrolabNeuralNetwork



origin_folder = '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/old_images/'
denoised_training_folder =  '/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/SDDS/images/'

#
# test_folder = "/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/images_test_rev1"
# denoised_test_folder = "/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/denoised_images_test"
#


origin_file_name_list = sorted(os.listdir(origin_folder))
# test_file_name_list = os.listdir(test_folder)

image_processor = AstrolabImageProcessor()

for file_name in origin_file_name_list:
    img = cv2.imread(os.path.join(origin_folder, file_name), cv2.IMREAD_UNCHANGED)
    img = image_processor.denoise(img)
    cv2.imwrite(os.path.join(denoised_training_folder, file_name), img)


#data = input_data.read_data_sets('MNIST_data', one_hot=True)
data = DataWrapper()
data.train.load_labels('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/labels.csv')
data.train.load_images_names('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/Galaxies/images/', input_dimension)

# neural_network = AstrolabNeuralNetwork()
# neural_network.create()

#
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
