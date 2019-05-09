from abstract_network import *


class SmallLayers:
    """ Definition of layers for a small ladder network """
    def __init__(self, network):
        self.network = network

    def inference0(self, input_x, class_labels, n_classes, is_training=True):
        with tf.variable_scope("inference0"):
            conv1 = conv2d_bn_lrelu(input_x, class_labels, n_classes, self.network.cs[1], [4, 4], 2, is_training, name='inference0/conv2d_1')
            conv2 = conv2d_bn_lrelu(conv1, class_labels, n_classes, self.network.cs[2], [4, 4], 2, is_training, name='inference0/conv2d_2')
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1 = tf.contrib.layers.fully_connected(conv2, self.network.cs[3], activation_fn=tf.identity)
            fc1 = scale_and_shift_flat(fc1, class_labels, n_classes, name='inference0/s_and_s_f_1')
            return fc1

    def ladder0(self, input_x, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder0"):
            conv1 = conv2d_bn_lrelu(input_x, class_labels, self.network.cs[1], [4, 4], 2, is_training, name='ladder0/conv2d_1')
            conv2 = conv2d_bn_lrelu(conv1, class_labels, self.network.cs[2], [4, 4], 2, is_training, name='ladder0/conv2d_2')
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.identity)
            fc1_mean = scale_and_shift_flat(fc1_mean, class_labels, name='ladder0/s_and_s_f_1')
            fc1_stddev = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.sigmoid)
            fc1_stddev = scale_and_shift_flat(fc1_stddev, class_labels, name='ladder0/s_and_s_f_2')
            return fc1_mean, fc1_stddev

    def inference1(self, latent1, class_labels, n_classes, is_training=True):
        with tf.variable_scope("inference1"):
            fc1 = fc_bn_lrelu(latent1, class_labels, n_classes, self.network.cs[3], is_training, name='inference1/fc_1')
            fc2 = fc_bn_lrelu(fc1, class_labels, n_classes, self.network.cs[3], is_training, name='inference1/fc_2')
            fc3 = tf.contrib.layers.fully_connected(fc2, self.network.cs[3], activation_fn=tf.identity)
            fc3 = scale_and_shift_flat(fc1, class_labels, n_classes, name='inference1/s_and_s_f_1')
            return fc3

    def ladder1(self, latent1, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder1"):
            fc1 = fc_bn_lrelu(latent1, class_labels, n_classes, self.network.cs[3], is_training, name='ladder1/fc_1')
            fc2 = fc_bn_lrelu(fc1, class_labels, n_classes, self.network.cs[3], is_training, name='ladder1/fc_2')
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder1_dim, activation_fn=tf.identity)
            fc3_mean = scale_and_shift_flat(fc3_mean, class_labels, n_classes, name='ladder1/s_and_s_f_1')
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder1_dim, activation_fn=tf.sigmoid)
            fc3_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='ladder1/s_and_s_f_2')
            return fc3_mean, fc3_stddev

    def ladder2(self, latent1, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder2"):
            fc1 = fc_bn_lrelu(latent1, class_labels, n_classes, self.network.cs[3], is_training, name='ladder2/fc_1')
            fc2 = fc_bn_lrelu(fc1, class_labels, n_classes, self.network.cs[3], is_training, name='ladder2/fc_2')
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder2_dim, activation_fn=tf.identity)
            fc3_mean = scale_and_shift_flat(fc3_mean, class_labels, n_classes, name='ladder2/s_and_s_f_1')
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder2_dim, activation_fn=tf.sigmoid)
            fc3_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='ladder2/s_and_s_f_2')
            return fc3_mean, fc3_stddev

    def combine_noise(self, latent, ladder, method='gated_add', name="default"):
        if method is 'concat':
            return tf.concat(values=[latent, ladder], axis=len(latent.get_shape())-1)
        else:
            if method is 'add':
                return latent + ladder
            elif method is 'gated_add':
                gate = tf.get_variable("gate", shape=latent.get_shape()[1:], initializer=tf.constant_initializer(0.1))
                tf.summary.histogram(name + "_noise_gate", gate)
                return latent + tf.multiply(gate, ladder)

    def generative0(self, latent1, ladder0=None, class_labels=None, n_classes=0, reuse=False, is_training=True):
        with tf.variable_scope("generative0") as gs:
            if reuse:
                gs.reuse_variables()

            if ladder0 is not None:
                ladder0 = fc_bn_lrelu(ladder0, class_labels, n_classes, n_classes, self.network.cs[3], name='generative0/fc_1')
                if latent1 is not None:
                    latent1 = self.combine_noise(latent1, ladder0, name="generative0")
                else:
                    latent1 = ladder0
            elif latent1 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent1, class_labels, n_classes, int(self.network.fs[2] * self.network.fs[2] * self.network.cs[2]), is_training, name='generative0/fc_2')
            fc1 = tf.reshape(fc1, tf.stack([tf.shape(fc1)[0], self.network.fs[2], self.network.fs[2], self.network.cs[2]]))
            conv1 = conv2d_t_bn_relu(fc1, class_labels, n_classes, self.network.cs[1], [4, 4], 2, is_training, name='generative0/conv2d_1')
            output = tf.contrib.layers.convolution2d_transpose(conv1, self.network.data_dims[-1], [4, 4], 2, activation_fn=tf.sigmoid)
            output = scale_and_shift(output, class_labels, n_classes, name='generative0/conv2d_2')
            output = (self.network.dataset.range[1] - self.network.dataset.range[0]) * output + self.network.dataset.range[0]
            return output

    def generative1(self, latent2, ladder1=None, class_labels=None, n_classes=0, reuse=False, is_training=True):
        with tf.variable_scope("generative1") as gs:
            if reuse:
                gs.reuse_variables()
            if ladder1 is not None:
                ladder1 = fc_bn_relu(ladder1, class_labels, n_classes, self.network.cs[3], is_training, name='generative1/fc_1')
                if latent2 is not None:
                    latent2 = self.combine_noise(latent2, ladder1, name="generative1")
                else:
                    latent2 = ladder1
            elif latent2 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent2, class_labels, n_classes, self.network.cs[3], is_training, name='generative1/fc_2')
            fc2 = fc_bn_relu(fc1, class_labels, n_classes, self.network.cs[3], is_training, name='generative1/fc_3')
            fc3 = tf.contrib.layers.fully_connected(fc2, self.network.cs[3], activation_fn=tf.identity)
            fc3 = scale_and_shift_flat(fc3, class_labels, n_classes, name='generative1/s_and_s_f_1')
            return fc3

    def generative2(self, latent3, ladder2, class_labels, n_classes=0, reuse=False, is_training=True):
        with tf.variable_scope("generative2") as gs:
            if reuse:
                gs.reuse_variables()
            fc1 = fc_bn_relu(ladder2, class_labels, n_classes, self.network.cs[3], is_training, name='generative2/fc_1')
            fc2 = fc_bn_relu(fc1, class_labels, n_classes, self.network.cs[3], is_training, name='generative2/fc_1')
            fc3 = tf.contrib.layers.fully_connected(fc2, self.network.cs[3], activation_fn=tf.identity)
            fc3 = scale_and_shift_flat(fc3, class_labels, n_classes, name='s_and_s_f_1')
            return fc3

