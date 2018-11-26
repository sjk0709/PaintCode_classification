# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:56:53 2017

@author: Jaekyung
"""
import sys, os
sys.path.append(os.pardir)  # parent directory
import tensorflow as tf
import numpy as np


tf.set_random_seed(777) # reproducibility


class Model:
    
    def __init__(self, sess, name, learning_rate=0.0001, feature_shape=[32,32,1], lable_size=12, weight_decay_rate=1e-5):
        self.sess = sess
        self._name = name
        self._learning_rate = learning_rate
        self._feature_shape = feature_shape
        self._lable_size = lable_size
                
        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay_rate)
        
        self._build_net()
        
        
        
    def _build_net(self):        
        with tf.variable_scope(self._name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing            
            self.training = tf.placeholder(tf.bool, name="training")

            # input place holders
            self.X = tf.placeholder( tf.float32, [None, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2]], name="input")           
            X_img = tf.reshape(self.X, [-1, self._feature_shape[0], self._feature_shape[1], self._feature_shape[2]])
            self.Y = tf.placeholder(tf.float32, [None, self._lable_size])
            
            # Convolutional Layer #1 and # Pooling Layer #1
            conv11 = tf.layers.conv2d(inputs=X_img, filters=64, kernel_size=[3,3], 
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)            # (?, W, H, 64)

            conv12 = tf.layers.conv2d(inputs=conv11, filters=64, kernel_size=[3,3], 
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)            # (?, W, H, 64)
            
            pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2,2],
                                            padding="SAME", strides=2)                      # (?, W/2, H/2, 64)
            
            dropout1 = tf.layers.dropout(inputs=pool1, 
                                         rate=0.7, training=self.training)
            
            # Convolutional Layer #2 and Pooling Layer #2
            conv21 = tf.layers.conv2d(inputs=dropout1, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu, 
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/2, H/2, 128)
            
            conv22 = tf.layers.conv2d(inputs=conv21, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/2, H/2, 128)
            
            pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2,2],
                                            padding="SAME", strides=2)                       # (?, W/4, H/4, 128)
            
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)
            
            # Convolutional Layer #2 and Pooling Layer #3
            conv31 = tf.layers.conv2d(inputs=dropout2, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/4, H/4, 256)
            
            conv32 = tf.layers.conv2d(inputs=conv31, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/4, H/4, 256)
            
            conv33 = tf.layers.conv2d(inputs=conv32, filters=256, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/4, H/4, 256)
            
            pool3 = tf.layers.max_pooling2d(inputs=conv33, pool_size=[2,2],
                                            padding="SAME", strides=2)                       # (?, W/8, H/8, 256)
            
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)
            
            # Convolutional Layer #2 and Pooling Layer #3
            conv41 = tf.layers.conv2d(inputs=dropout3, filters=128, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/8, H/8, 128)
            
            conv42 = tf.layers.conv2d(inputs=conv41, filters=64, kernel_size=[3,3],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, W/8, H/8, 64)
            
            pool4 = tf.layers.max_pooling2d(inputs=conv42, pool_size=[2,2],
                                            padding="SAME", strides=2)                       # (?, W/16, H/16, 64)
            
            dropout4 = tf.layers.dropout(inputs=pool4,
                                         rate=0.7, training=self.training)
            # Dense Layer with Relu ========================================================================
            flat = tf.reshape(dropout4, [-1, 1, 1, int(self._feature_shape[0]/16)*int(self._feature_shape[1]/16)*64])  # (?, 1, 1,  W/8* H/8*256)
                              
            conv41 = tf.layers.conv2d(inputs=flat, filters=1024, kernel_size=[1,1],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, 1, 1, 1024)
            
            conv42 = tf.layers.conv2d(inputs=conv41, filters=self._lable_size, kernel_size=[1,1],
                                     padding="SAME", activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=self.kernel_regularizer)             # (?, 1, 1,  label_size)
 
            self.logits = tf.reshape(conv42, [-1, self._lable_size])    # (?, label_size)
            
                    
        self.prob = tf.nn.softmax(self.logits, name="prob")
        self.result = tf.argmax(self.logits, 1, name="result")
            
        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.prob,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})
