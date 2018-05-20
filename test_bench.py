import tensorflow as tf
import numpy as np


x_data = np.random.randn(2000,3)
w_real = [0.4, 0.6, 0.2]
b_real = -0.3
y_data = np.matmul(w_real, x_data.T)+b_real

num_iters = 10
g = tf.Graph()
wb = []

with g.as_default():
   x = tf.placeholder(tf.float32, shape=[None, 3])
   y_true = tf.placeholder(tf.float32, shape=None)

   with tf.name_scope('inference') as scope:
      w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='W')
      b = tf.Variable(0, dtype=tf.float32, name='b')
      y_pred = tf.matmul(w, tf.transpose(x)) + b

   with tf.name_scope('loss') as scope:
      loss = tf.reduce_mean(tf.square(y_true - y_pred))

   with tf.name_scope('training') as scope:
      lr = 0.5
      optimizer = tf.train.GradientDescentOptimizer(lr)
      train = optimizer.minimize(loss)

   init = tf.global_variables_initializer()

   with tf.Session() as sess:
      sess.run(init)
      for step in range(num_iters):
         sess.run(train, {x: x_data, y_true: y_data})
         if (step % 5 == 0):
            print(step, sess.run([w, b]))
            wb.append(sess.run([w, b]))

      print(10, sess.run([w, b]))