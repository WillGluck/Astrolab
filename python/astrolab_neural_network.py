import tensorflow as tf
from loader import MNIST

from tensorflow.examples.tutorials.mnist import input_data

class AstrolabNeuralNetwork:

    def __init__(self):
        self.session = tf.InteractiveSession()

    # def test(self):
    #
    #     input_dimension = 784
    #     output_dimension = 10
    #
    #     #BatchSize and Image size
    #     x = tf.placeholder(tf.float32, shape=[None, input_dimension])
    #     #Output size
    #     correct_y = tf.placeholder(tf.float32, shape=[None, output_dimension])
    #     #Weight
    #     W = tf.Variable(tf.zeros([input_dimension, output_dimension]))
    #     #bias
    #     b = tf.Variable(tf.zeros([output_dimension]))
    #     #initialize variables on session
    #     self.session.run(tf.initialize_all_variables())

    def create(self):

        #INIT

        input_dimension = 784
        output_dimension = 10

        #BatchSize and Image size
        x = tf.placeholder(tf.float32, shape=[None, input_dimension])
        #Output size
        correct_y = tf.placeholder(tf.float32, shape=[None, output_dimension])

        #LOAD MNIST
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        #TODO

        #FIRST LAYER

        #5x5 - patch size, 1 - input channels, 32 - output channels
        conv1_W = self.create_random_weights([5, 5, 1, 32])
        #32 - output channels
        conv1_b = self.create_random_biases([32])
        #reshape for tensor
        x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
        #Convolv and apply bias.
        conv1_o = tf.nn.relu(self.do_conv2d(x_reshaped, conv1_W) + conv1_b)
        pool1_o = self.do_max_pool_2x2(conv1_o)

        #SECOND LAYER - Repeat convolution and max pooling

        conv2_W = self.create_random_weights([5, 5, 32, 64])
        conv2_b = self.create_random_biases([64])

        conv2_o = tf.nn.relu(self.do_conv2d(pool1_o, conv2_W) + conv2_b)
        pool2_o = self.do_max_pool_2x2(conv2_o)

        #THIRD LAYER - Connected layer

        d1_W = self.create_random_weights([7 * 7 * 64, 1024])
        d1_b = self.create_random_biases([1024])

        pool2_o_reshaped = tf.reshape(pool2_o, [-1, 7 * 7 * 64])
        d1_o = tf.nn.relu(tf.matmul(pool2_o_reshaped, d1_W) + d1_b)

        #Dropout output to avoid overfitting
        resistence = tf.placeholder(tf.float32)
        d1_o_dropout = tf.nn.dropout(d1_o, resistence)

        #FINAL LAYER - Softmax

        d2_W = self.create_random_weights([1024, 10])
        d2_b = self.create_random_biases([10])

        predicted_y = tf.nn.softmax(tf.matmul(d1_o_dropout, d2_W) + d2_b)

        #TRAIN

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_y * tf.log(predicted_y), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(correct_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.session.run(tf.initialize_all_variables())
        for i in range(1000):

            batch = mnist.train.next_batch(50)

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict = {
                    x:batch[0],
                    correct_y:batch[1],
                    resistence:0.5
                })
                print("Step %d, training accuracy %g"%(i, train_accuracy))

            train_step.run(feed_dict={
                x:batch[0],
                correct_y:batch[1],
                resistence:0.5
            })

        print("Test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images,
            correct_y: mnist.test.labels,
            resistence:1.0
        }))

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
