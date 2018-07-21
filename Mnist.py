import tensorflow as tf
import numpy as np
import scipy
from tensorflow.examples.tutorials.mnist import input_data

"""
Implemented with Tensorflow

Total of 2 layers because I'm lazy
"""

def train():
    #setup stuff
    input_layer = 784
    output_layer = 10
    learning_rate = 0.5
    batch_size = 100

    global mnist
    mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

    #Define the network Input Layer
    #None means that it has dimensions any length X 784
    global network_input
    network_input = tf.placeholder(tf.float32, [None, input_layer])
    #Define the output Layer
    global target_output
    target_output = tf.placeholder(tf.float32, [None, output_layer])

    #Compute the net output of the input layer
    weights = tf.Variable(tf.zeros([input_layer, output_layer]))
    biases = tf.Variable(tf.zeros([10]))
    global network_output
    network_output = tf.nn.softmax(tf.matmul(network_input, weights) + biases)
    
    #calculate our loss function    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(target_output * tf.log(network_output), reduction_indices=[1]))

    #Gradient Descent + BackPropagation
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    #Yay! We can begin training our network

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict={network_input: batch_xs, target_output: batch_ys})

    return sess

def Mnist(data, sess):
    #Test our Network
    #correct_prediction = tf.equal(tf.argmax(network_output,1), tf.argmax(target_output,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={network_input: mnist.test.images, target_output: mnist.test.labels}))

    data = (np.array(data))
    #print(type(data))
    result = sess.run(tf.argmax(network_output,1),
                      feed_dict={network_input: [data]})
    
    return ' '.join(map(str, result))

