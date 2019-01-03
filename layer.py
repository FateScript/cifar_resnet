import tensorflow as tf
import numpy as np
from IPython import embed
from config import *


def approx_relu(x):
    return tf.log(1 + tf.exp(x))

#tf.set_random_seed(3)

def create_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(seed=3), is_fc_layer=False ):
    ans = 1
    for i in shape:
        ans *= i
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    var = tf.get_variable(name, shape=[ans], initializer=initializer, regularizer=regularizer)
    #reshape_var = tf.reshape(var, shape=[-1])
    #return reshape_var
    return var

'''
def weights_variable(shape, name, norm=True, weight_decay=0.0001):
    tf.set_random_seed(3)
    #initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    weight = tf.Variable(initial(shape), name=name, collections=['variables'])
    #weight = tf.Variable(initial, name=name, collections=['variables'])
    #weight = tf.get_variable(name, initializer=initial, collections=['weights'])
    if norm:
        weight_norm = tf.reduce_sum(
            input_tensor = weight_decay * tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]), name='weight_norm'
        )
        tf.add_to_collection('losses', weight_norm)

    return weight

def bias_variable(shape, name):
    init = tf.constant(0.1, shape=shape)
    bias = tf.Variable(init, name=name)
    return bias
'''

def conv(inputs, filter_shape, strides, name):
    filters = create_variable(name='conv', shape=filter_shape)
    x = tf.nn.conv2d(inputs, filters, strides=[1, strides, strides, 1], padding='SAME')
    return x


def maxpool(inputs, win_size, strides, name, padding="SAME"):
    output = tf.nn.max_pool(inputs, ksize=[1, win_size, win_size, 1], strides=[1, strides, strides, 1],padding=padding)
    return output

def avgpool(inputs, win_size, strides, name, padding="SAME"):
    output = tf.nn.avg_pool(inputs, ksize=[1, win_size, win_size, 1], strides=[1, strides, strides, 1], padding=padding)
    return output

def fc(inputs, output_dim, name):
    input_dim = inputs.get_shape().as_list()[-1]
    reshaped_weights = create_variable(name=name+'_weights', shape=[input_dim, output_dim], initializer=tf.initializers.variance_scaling(distribution='uniform'))
    weights = tf.reshape(reshaped_weights, shape=[input_dim, output_dim])
    bias = create_variable(name=name+'_bias', shape=[output_dim], initializer=tf.zeros_initializer())
    output = tf.matmul(inputs, weights) + bias
    #output = tf.matmul(inputs, weights)
    return output


def batchNorm(inputs, is_training, name, decay=0.9, bn_epsilon=1e-5):
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
    reshaped_filters = create_variable(name='conv', shape=filter_shape)
    filters = tf.reshape(reshaped_filters, shape=filter_shape)
    x = tf.nn.conv2d(inputs, filters, strides=[1, strides, strides, 1], padding='SAME')
    x = batchNorm(x, is_training, name)
    #output = tf.nn.relu(x)
    output = approx_relu(x)
    return output


def bn_relu_conv(inputs, filter_shape, strides, is_training, name):
    x = batchNorm(inputs, is_training, name)
    #x = tf.nn.relu(x)
    x = approx_relu(x)
    reshaped_filters = create_variable(name='conv', shape=filter_shape)
    filters = tf.reshape(reshaped_filters, shape=filter_shape)
    output = tf.nn.conv2d(x, filters, strides=[1, strides, strides, 1], padding='SAME')
    return output
    

def res_block(inputs, output_dim, is_training, name, first_block=False):
    input_dim = inputs.get_shape().as_list()[-1]
    if input_dim < output_dim:
        strides = 2
        reshaped_filters = create_variable(name=name+'_shortcut', shape=[1, 1, input_dim, output_dim])
        filters = tf.reshape(reshaped_filters, shape=[1,1,input_dim,output_dim])
        short_cut = tf.nn.conv2d(inputs, filters, strides=[1, 2, 2, 1], padding='SAME')
        #x = tf.nn.avg_pool(inputs, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
        #short_cut = tf.pad(x, [[0,0],[0,0],[0,0], [input_dim//2,input_dim//2]])
    else:
        strides = 1
        short_cut = inputs
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            #embed(header='conv1_in_res_block')
            reshaped_filters = create_variable('conv', [3,3, input_dim, output_dim])
            filters = tf.reshape(reshaped_filters, shape=[3,3,input_dim,output_dim])
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
