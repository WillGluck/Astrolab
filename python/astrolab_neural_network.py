import tensorflow as tf
from data import DataWrapper
from matplotlib import pyplot as plt
import numpy as np
import os

class AstrolabNeuralNetwork:

    def __init__(self, input_shape, input_channels, output_dimension):

        self.input_shape = input_shape
        self.saved_variables_path = "./saved_variables/" + str(input_shape) + "_3/trained_variables.ckpt"
        # self.saved_variables_path = "./saved_variables/"
        self.input_channels = input_channels
        self.input_shape_conved = int(self.input_shape / 4)
        self.input_dimension = self.input_shape * self.input_shape
        self.output_dimension = output_dimension

        self.session = tf.InteractiveSession()
        self.x = None
        self.resistence = None
        self.predicted_y = None
        self.correct_y = None
        self.accuracy = None
        self.train_step = None
        self.cross_entropy = None

        self.conv2_W = None

        self.load_graph()


    def classify(self, image):

        if os.path.isfile(self.saved_variables_path):

            saver = tf.train.Saver()
            saver.restore(self.session, self.saved_variables_path)

            classification =  self.session.run(tf.argmax(self.predicted_y, 1), feed_dict={self.x : image, self.resistence:1.0})

            return classification


    def train(self, data_wrapper, batch_size, iterations):

        self.session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        test = data_wrapper.test.next_batch(978)

        for i in range(iterations):

            batch = data_wrapper.train.next_batch(batch_size)
            # print(batch[1][0])

            if i % 100 == 0:

                train_accuracy = self.accuracy.eval(feed_dict = {
                    self.x: batch[0],
                    self.correct_y:batch[1],
                    self.resistence:1
                })
                print("\nStep %d\nTraining accuracy %g"%(i, train_accuracy))

            if i % 1000 == 0:
                print("Test accuracy %g"%self.accuracy.eval(feed_dict={
                    self.x: test[0],
                    self.correct_y: test[1],
                    self.resistence:1.0
                }))

                saver.save(self.session, self.saved_variables_path)
                print("Saving variables")


            _, loss_val = self.session.run([self.train_step, self.cross_entropy], feed_dict={
                self.x: batch[0],
                self.correct_y:batch[1],
                self.resistence:0.5
            })

            if i % 100 == 0:
                print("Loss: " + str(loss_val))
            #     print("\n\n\n\n\n")

        # saver.save(self.session, self.saved_variables_path)


    def load_graph(self):

        #BatchSize and Image size
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dimension])
        #Output size
        self.correct_y = tf.placeholder(tf.float32, shape=[None, self.output_dimension])

        #FIRST LAYER

        #5x5 - patch size, 1 - input channels, 32 - output channels
        conv1_W = self.create_random_weights([5, 5, self.input_channels, 32])
        #32 - output channels
        conv1_b = self.create_random_biases([32])
        #reshape for tensor
        x_reshaped = tf.reshape(self.x, [-1, self.input_shape, self.input_shape, self.input_channels])
        #Convolv and apply bias.
        conv1_o = tf.nn.relu(self.do_conv2d(x_reshaped, conv1_W) + conv1_b)
        pool1_o = self.do_max_pool_2x2(conv1_o)

        #SECOND LAYER - Repeat convolution and max pooling

        self.conv2_W = self.create_random_weights([5, 5, 32, 64])
        conv2_b = self.create_random_biases([64])

        conv2_o = tf.nn.relu(self.do_conv2d(pool1_o, self.conv2_W) + conv2_b)
        pool2_o = self.do_max_pool_2x2(conv2_o)


        # conv3_W = self.create_random_weights([5, 5, 64, 128])
        # conv3_b = self.create_random_biases([128])
        #
        # conv3_o = tf.nn.relu(self.do_conv2d(pool2_o, conv3_W) + conv3_b)
        # pool3_o = self.do_max_pool_2x2(conv3_o)
        #
        # conv4_W = self.create_random_weights([5, 5, 128, 256])
        # conv4_b = self.create_random_biases([256])
        #
        # conv4_o = tf.nn.relu(self.do_conv2d(pool3_o, conv4_W) + conv4_b)
        # pool4_o = self.do_max_pool_2x2(conv4_o)

        #THIRD LAYER - Connected layer

        filter_count = 64
        neurons_count = 1024

        d1_W = self.create_random_weights([self.input_shape_conved * self.input_shape_conved * filter_count, neurons_count])
        d1_b = self.create_random_biases([neurons_count])

        #TODO mudar de pool2 para pool3
        pool2_o_reshaped = tf.reshape(pool2_o, [-1, self.input_shape_conved * self.input_shape_conved * filter_count])
        d1_o = tf.nn.relu(tf.matmul(pool2_o_reshaped, d1_W) + d1_b)


        # d2_W = self.create_random_weights([1024, 1024])
        # d2_b = self.create_random_biases([1024])
        # d2_o = tf.nn.relu(tf.matmul(d1_o, d2_W) + d2_b)

        #Dropout output to avoid overfitting
        self.resistence = tf.placeholder(tf.float32)
        # TODO mudar de d1 para d2
        d1_o_dropout = tf.nn.dropout(d1_o, self.resistence)

        #FINAL LAYER - Softmax

        d2_W = self.create_random_weights([neurons_count, self.output_dimension])
        d2_b = self.create_random_biases([self.output_dimension])

        self.predicted_y = tf.nn.softmax(tf.matmul(d1_o_dropout, d2_W) + d2_b)

        #TRAIN
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predicted_y, self.correct_y))
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.correct_y * tf.log(tf.clip_by_value(self.predicted_y, 1e-10,1.0))))
        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.correct_y * tf.log(self.predicted_y)))
        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.correct_y * tf.log(self.predicted_y), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.predicted_y, 1), tf.argmax(self.correct_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
