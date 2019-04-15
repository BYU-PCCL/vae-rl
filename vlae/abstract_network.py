import tensorflow as tf
import numpy as np
import math
import glob
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import os, sys, shutil, re
from CoordConv import AddCoords, CoordConv
from tensorflow.python.ops import math_ops

def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)

conv2d = tf.contrib.layers.convolution2d
conv2d_t = tf.contrib.layers.convolution2d_transpose
fc_layer = tf.contrib.layers.fully_connected
initializer = tf.random_normal_initializer(stddev=0.02)

def conditional_instance_norm(inputs, scope_bn, labels):
    beta = tf.get_variable(name=scope_bn+'beta', shape=[labels.shape[-1], inputs.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)
    gamma = tf.get_variable(name=scope_bn+'gamma', shape=[labels.shape[-1], inputs.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True)
    beta = tf.matmul(labels, beta)
    gamma = tf.matmul(labels, gamma)

    shape = inputs.get_shape().as_list()
    if len(shape) == 4:
         beta = tf.reshape(beta, [-1, 1, 1, shape[-1]])
         gamma = tf.reshape(gamma, [-1, 1, 1, shape[-1]])
    return gamma * inputs + beta

def conv2d_bn_lrelu(inputs, num_outputs, kernel_size, stride, classes, name, is_training=True, add_coords=False):
    if add_coords:
        _, x_dim, y_dim, _ = inputs.get_shape().as_list()
        conv = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=False, skiptile=True)(inputs)
        conv = tf.contrib.layers.convolution2d(conv, num_outputs, kernel_size, stride, weights_initializer=initializer,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn=tf.identity)
    else:
        conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride, weights_initializer=initializer,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = conditional_instance_norm(conv, name, classes)
    conv = lrelu(conv)
    return conv

def conv2d_t_bn_relu(inputs, num_outputs, kernel_size, stride, classes, name, is_training=True, add_coords=False):
    if add_coords:
        _, x_dim, y_dim, _ = inputs.get_shape().as_list()
        conv = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=False, skiptile=True)(inputs)
        conv = tf.contrib.layers.convolution2d_transpose(conv, num_outputs, kernel_size, stride, weights_initializer=initializer,
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn=tf.identity)
    else:
        conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride, weights_initializer=initializer,
                                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(conv, is_training=is_training)
    conv = conditional_instance_norm(conv, name, classes)
    conv = lrelu(conv)
    return conv

def fc_bn_lrelu(inputs, num_outputs, classes, name, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=initializer,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = conditional_instance_norm(fc, name, classes)
    fc = lrelu(fc)
    return fc


def fc_bn_relu(inputs, num_outputs, classes, name, is_training=True):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=initializer,
                                           weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                           activation_fn=tf.identity)
    fc = tf.contrib.layers.batch_norm(fc, is_training=is_training)
    fc = conditional_instance_norm(fc, name, classes)
    fc = tf.nn.relu(fc)
    return fc


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


class Network:
    def __init__(self, dataset, batch_size, file_path):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_path = file_path
        self.learning_rate_placeholder = tf.placeholder(shape=[], dtype=tf.float32, name="lr_placeholder")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        self.name = "default"

        self.iteration = 0
        self.learning_rate = 0.0
        self.read_only = False

        self.do_generate_samples = False
        self.do_generate_conditional_samples = False
        self.do_generate_manifold_samples = False

    def make_model_path(self):
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        if not os.path.isdir(self.file_path + self.name):
            os.mkdir(self.file_path + self.name)

    def print_network(self):
        self.make_model_path()
        if os.path.isdir(self.file_path + self.name):
            for f in os.listdir(self.file_path + self.name):
                if re.search(r"events.out*", f):
                    os.remove(os.path.join(self.file_path + self.name, f))
        self.writer = tf.summary.FileWriter(self.file_path + self.name, self.sess.graph)
        self.writer.flush()

    def save_network(self):
        if not self.read_only:
            with tf.device('/cpu:0'):
                saver = tf.train.Saver()
            self.make_model_path()
            file_name = self.file_path + self.name + "/" + self.name + ".ckpt"
            if os.path.isfile(file_name):
                if not os.path.isdir(self.file_path + "old"):
                    os.mkdir(self.file_path + "old")
                os.rename(file_name, self.file_path + "old/" + self.name + ".ckpt")
            saver.save(self.sess, file_name)

    def init_network(self, restart=False):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if restart:
            return
        file_name = self.file_path + self.name + "/" + self.name + ".ckpt"
        if len(glob.glob(file_name + '*')) != 0:
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, file_name)
                print("Successfully restored model")
            except:
                print("Warning: network load failed, reinitializing all variables", sys.exc_info()[0])
                self.sess.run(tf.global_variables_initializer())
        else:
            print("No checkpoint file found, Initializing model from random")

    def train(self, batch_input, batch_target, labels=None):
        return None

    def test(self, batch_input, labels=None):
        return None
