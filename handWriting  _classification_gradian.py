
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST", one_hot=True)

num_iters = 100
minibatch_size = 100


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(num_iters):
        batch_xs, batch_ys = data.train.next_batch(minibatch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y_true: batch_ys})

    testing = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print('Accuracy: {:.4}%'.format(testing * 100))