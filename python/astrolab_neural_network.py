import tensorflow as tf
from data import DataWrapper
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class AstrolabNeuralNetwork:

    def __init__(self):
    
        self.session = tf.InteractiveSession()
        self.predicted_y = None        
        
        self.input_shape = 112
        self.input_shape_conved = int(self.input_shape / 4)
        self.input_dimension = self.input_shape * self.input_shape
        self.output_dimension = 3
        
    def load_graph(self, image):
        print("temp")
        
    
    def classify(self, image):
        print("test")       
        
        saver = tf.train.Saver()
        saver.restore(self.session, "./trained_variables.ckpt")
         
        classification =  sess.run(tf.argmax(self.predicted_y, 1), feed_dict=[x:image])
        classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
        
        plt.imshow(image.reshape(112, 112), cmap=plt.cm.binary)
        plt.show()
        
        print 'NN predicted', classification[0]


    def train(self, data_wrapper):

        #BatchSize and Image size
        x = tf.placeholder(tf.float32, shape=[None, self.input_dimension])
        #Output size
        correct_y = tf.placeholder(tf.float32, shape=[None, self.output_dimension])

        #LOAD MNIST
        
        training_images = None
        test_images = None
        training_labels = None
        test_labels = None

        #TODO

        #FIRST LAYER

        #5x5 - patch size, 1 - input channels, 32 - output channels
        conv1_W = self.create_random_weights([5, 5, 1, 32])
        #32 - output channels
        conv1_b = self.create_random_biases([32])
        #reshape for tensor
        x_reshaped = tf.reshape(x, [-1, self.input_shape, self.input_shape, 1])
        #Convolv and apply bias.
        conv1_o = tf.nn.relu(self.do_conv2d(x_reshaped, conv1_W) + conv1_b)
        pool1_o = self.do_max_pool_2x2(conv1_o)

        #SECOND LAYER - Repeat convolution and max pooling

        conv2_W = self.create_random_weights([5, 5, 32, 64])
        conv2_b = self.create_random_biases([64])

        conv2_o = tf.nn.relu(self.do_conv2d(pool1_o, conv2_W) + conv2_b)
        pool2_o = self.do_max_pool_2x2(conv2_o)

        #THIRD LAYER - Connected layer

        d1_W = self.create_random_weights([self.input_shape_conved * self.input_shape_conved * 64, 1024])
        d1_b = self.create_random_biases([1024])

        pool2_o_reshaped = tf.reshape(pool2_o, [-1, self.input_shape_conved * self.input_shape_conved * 64])
        d1_o = tf.nn.relu(tf.matmul(pool2_o_reshaped, d1_W) + d1_b)

        #Dropout output to avoid overfitting
        resistence = tf.placeholder(tf.float32)
        d1_o_dropout = tf.nn.dropout(d1_o, resistence)

        #FINAL LAYER - Softmax

        d2_W = self.create_random_weights([1024, self.output_dimension])
        d2_b = self.create_random_biases([self.output_dimension])

        self.predicted_y = tf.nn.softmax(tf.matmul(d1_o_dropout, d2_W) + d2_b)

        #TRAIN

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_y * tf.log(self.predicted_y), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.predicted_y, 1), tf.argmax(correct_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        for i in range(20000):

            batch = data_wrapper.train.next_batch(10)

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict = {
                    x: batch[0],
                    correct_y:batch[1],
                    resistence:0.5
                })
                print("Step %d, training accuracy %g"%(i, train_accuracy))

            train_step.run(feed_dict={
                x: batch[0],
                correct_y:batch[1],
                resistence:0.5
            })

        #saver.restore(self.session, "./trained_variables.ckpt")
        batch = data_wrapper.train.next_batch(1000)

        print("Test accuracy %g"%accuracy.eval(feed_dict={
            x: batch[0],
            correct_y: batch[1],
            resistence:1.0
        }))

        saver.save(self.session, "./trained_variables.ckpt")

    def create_random_weights(self, shape):
        random_weights = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(random_weights)

    def create_random_biases(self, shape):
        random_biases = tf.constant(0.1, shape=shape)
        return tf.Variable(random_biases)

    def do_conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def do_max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
