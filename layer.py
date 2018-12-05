import tensorflow as tf
import numpy as np
from IPython import embed
from config import *


#tf.set_random_seed(3)

def create_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(seed=3), is_fc_layer=False ):
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer)
    return var


def conv(inputs, filter_shape, strides, name):
    with tf.name_scope(name) as scope:
        filters = create_variable(name='conv', shape=filter_shape)
        x = tf.nn.conv2d(inputs, filters, strides=[1, strides, strides, 1], padding='SAME')
        return x


def maxpool(inputs, win_size, strides, name, padding="SAME"):
    with tf.name_scope(name) as scope:
        output = tf.nn.max_pool(inputs, ksize=[1, win_size, win_size, 1], strides=[1, strides, strides, 1],padding=padding)
        return output


def avgpool(inputs, win_size, strides, name, padding="SAME"):
    with tf.name_scope(name) as scope:
        output = tf.nn.avg_pool(inputs, ksize=[1, win_size, win_size, 1], strides=[1, strides, strides, 1], padding=padding)
        return output


def fc(inputs, output_dim, name):
    with tf.name_scope(name) as scope:
        input_dim = inputs.get_shape().as_list()[-1]
        weights = create_variable(name=name+'_weights', shape=[input_dim, output_dim], initializer=tf.initializers.variance_scaling(distribution='uniform'))
        bias = create_variable(name=name+'_bias', shape=[output_dim], initializer=tf.zeros_initializer())
        output = tf.matmul(inputs, weights) + bias
        return output


def batchNorm(inputs, is_training, name, decay=0.9, bn_epsilon=1e-5):
    with tf.name_scope(name) as scope:
        shape_list = inputs.get_shape().as_list()
        param_shape = shape_list[-1]
        axis = list( range(len(shape_list)-1) )
        batch_mean, batch_var = tf.nn.moments(inputs, axis)
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        beta = tf.get_variable(name+'_beta', param_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        gamma = tf.get_variable(name+'_gamma', param_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        def mean_var_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(is_training, mean_var_update, lambda: (ema.average(batch_mean), ema.average(batch_var)) )
        output = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, bn_epsilon)
        return output


def conv_bn_relu(inputs, filter_shape, strides, is_training, name):
    with tf.name_scope(name) as scope:
        filters = create_variable(name='conv', shape=filter_shape)
        x = tf.nn.conv2d(inputs, filters, strides=[1, strides, strides, 1], padding='SAME')
        x = batchNorm(x, is_training, name)
        output = tf.nn.relu(x)
        return output


def bn_relu_conv(inputs, filter_shape, strides, is_training, name):
    with tf.name_scope(name) as scope:
        x = batchNorm(inputs, is_training, name)
        x = tf.nn.relu(x)
        filters = create_variable(name='conv', shape=filter_shape)
        output = tf.nn.conv2d(x, filters, strides=[1, strides, strides, 1], padding='SAME')
        return output


def res_block(inputs, output_dim, is_training, name, first_block=False):
    with tf.name_scope(name) as scope:
        input_dim = inputs.get_shape().as_list()[-1]
        if input_dim < output_dim:
            strides = 2
            filters = create_variable(name=name+'_shortcut', shape=[1, 1, input_dim, output_dim])
            short_cut = tf.nn.conv2d(inputs, filters, strides=[1, 2, 2, 1], padding='SAME')
            #Here is another way to implement shortcut
            #x = tf.nn.avg_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
            #short_cut = tf.pad(x, [[0,0],[0,0],[0,0], [input_dim//2,input_dim//2]])
        else:
            strides = 1
            short_cut = inputs
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filters = create_variable('conv', [3,3, input_dim, output_dim])
                x = tf.nn.conv2d(inputs, filters, strides=[1,1,1,1], padding='SAME')
            else:
                x = bn_relu_conv(inputs, [3, 3, input_dim, output_dim], strides, is_training, name="conv_bn_relu1")
        with tf.variable_scope('conv2_in_block'):
            x = bn_relu_conv(x, [3, 3, output_dim, output_dim], 1, is_training, name="conv_bn_relu1")
        #if(output_dim==32):
        #    embed(header='dim32')
        return x + short_cut


def res_bottleneck(inputs):
    pass
