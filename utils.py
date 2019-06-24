import numpy as np
import tensorflow as tf

def conv_2d(input_x ,kernel,strides=[1, 1, 1, 1],padding='SAME'):
    weights= tf.Variable(tf.truncated_normal(kernel, stddev=0.05))
    biases =  tf.Variable(tf.constant(0.05, shape=[kernel[-1]]))
    return tf.nn.conv2d(input=input_x,filter=weights,strides=strides,padding=padding)+biases

def maxpool(input_x,kernel,strides=[1,2,2,1],padding="SAME"):
    return tf.nn.max_pool(value=input_x, ksize=kernel, strides=strides,padding=padding)

def flatten(input_x):
    return tf.layers.flatten(input_x)

def dense(input_x,num_inputs,num_outputs,name):
    weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    return tf.add(tf.matmul( input_x, weights), biases , name=name)

def dropout(input_x , keep_prob):
    return tf.nn.dropout(input_x,keep_prob=keep_prob)

def relu(input_x):
    return tf.nn.relu(input_x)

def softmax(logits):
    return tf.nn.softmax(logits)

def normalize_inputs(inputs):
        pixel_depth = 255.0
        return (inputs - (pixel_depth / 2)) / (pixel_depth / 2)