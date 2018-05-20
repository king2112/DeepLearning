'''---------------------------------------------------------------------
-                                                                   -
-                        Simple network by tensorflow (1)           -
-                                                                   -
   ---------------------------------------------------------------------'''


import tensorflow as tf
import numpy as np


x = tf.placeholder(dtype=tf.float32 , name= 'X' , shape=(4,9))
w = tf.placeholder(dtype=tf.float32 , name= 'W' , shape=(9,1))
b = tf.fill((4,1),-1, name='bias' )
b = tf.cast(b , tf.float32)
y = tf.matmul(x,w)+b
s = tf.reduce_max(y)

x_data = np.random.randn(4,9)
w_data = np.random.randn(9,1)

with tf.Session() as sess:
   out= sess.run(s , feed_dict={x : x_data , w: w_data})

print(" output : {}". format(out))

