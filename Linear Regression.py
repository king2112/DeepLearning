'''---------------------------------------------------------------------
-                                                                       -
-                        Simple network by tensorflow (2)               -
-                        (     Linear Regression      )                 -
   ---------------------------------------------------------------------'''


# inputs are : X , Y , W , B

import tensorflow as tf
import numpy as np


x_data = np.random.randn(2000 ,3)
w_real = [0.4 , 0.6 , 0.2]
b_real = 0.2
y_data = np.matmul(w_real, x_data.T)+b_real



#######################################################
g = tf.Graph()
wb= []
num_itrs:10

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='X')
    y_true = tf.placeholder(dtype=tf.float32, shape=None)




    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype = tf.float32 , name = 'w')
        b = tf.Variable( 0  , dtype = tf.float32  , name = 'b')
        y_perd = tf.matmul(w ,tf.transpose(x))+b





    with g.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_perd))




    with g.name_scope("training") as scope:
        learning_rate= 0.5

        optimzer=tf.train.GradientDescentOptimizer(learning_rate)
        train = optimzer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for steps in range(100):
            sess.run(train , feed_dict={y_true:y_data , x:x_data})
            if(steps%5==0):
                print(steps , sess.run([w, b] ))
                wb.append(sess.run([w, b]))

        print(10 , sess.run([w, b]))


