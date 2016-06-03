import os
import cv2
import tensorflow
from astrolab_image_processor import AstrolabImageProcessor
from astrolab_neural_network import AstrolabNeuralNetwork

img = cv2.imread('/media/willgluck/a2aa6a5f-a88a-45c7-af45-d38ccf2b7639/work/100008.jpg', cv2.IMREAD_UNCHANGED)
image_processor = AstrolabImageProcessor()
img = image_processor.denoise(img)

neural_network = AstrolabNeuralNetwork()
neural_network.create()

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
