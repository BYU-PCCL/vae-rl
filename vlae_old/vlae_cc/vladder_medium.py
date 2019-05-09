from abstract_network import *


class MediumLayers:
    """ Definition of layers for a medium sized ladder network """
    def __init__(self, network):
        self.network = network

    def inference0(self, input_x, class_labels, n_classes, is_training=True):
        with tf.variable_scope("inference0"):
            conv1 = conv2d_bn_lrelu(input_x, class_labels, n_classes, self.network.cs[1], [4, 4], 2, is_training, name='inference0/conv2d_1')
            conv2 = conv2d_bn_lrelu(conv1, class_labels, n_classes, self.network.cs[1], [4, 4], 1, is_training, name='inference0/conv2d_2')
            return conv2

    def ladder0(self, input_x, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder0"):
            conv1 = conv2d_bn_lrelu(input_x, class_labels, n_classes, self.network.cs[1], [4, 4], 2, is_training, name='inference1/conv2d_1')
            conv2 = conv2d_bn_lrelu(conv1, class_labels, n_classes, self.network.cs[1], [4, 4], 1, is_training, name='inference1/conv2d_2')
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.identity)
            fc1_mean = scale_and_shift_flat(fc1_mean, class_labels, n_classes, name='inference1/s_and_s_f_1')
            fc1_stddev = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.sigmoid)
            fc1_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='inference1/s_and_s_f_2')
            return fc1_mean, fc1_stddev

    def inference1(self, latent1, class_labels, n_classes, is_training=True):
        with tf.variable_scope("inference1"):
            conv3 = conv2d_bn_lrelu(latent1, class_labels, n_classes, self.network.cs[2], [4, 4], 2, is_training, name='inference1/conv2d_1')
            conv4 = conv2d_bn_lrelu(conv3, class_labels, n_classes, self.network.cs[2], [4, 4], 1, is_training, name='inference1/conv2d_2')
            return conv4

    def ladder1(self, latent1, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder1"):
            conv3 = conv2d_bn_lrelu(latent1, class_labels, n_classes, self.network.cs[2], [4, 4], 2, is_training, name='ladder1/conv2d_1')
            conv4 = conv2d_bn_lrelu(conv3, class_labels, n_classes, self.network.cs[2], [4, 4], 1, is_training, name='ladder1/conv2d_2')
            conv4 = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
            fc1_mean = tf.contrib.layers.fully_connected(conv4, self.network.ladder1_dim, activation_fn=tf.identity)
            fc1_mean = scale_and_shift_flat(fc1_mean, class_labels, n_classes, name='ladder1/s_and_s_f_1')
            fc1_stddev = tf.contrib.layers.fully_connected(conv4, self.network.ladder1_dim, activation_fn=tf.sigmoid)
            fc1_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='ladder1/s_and_s_f_2')
            return fc1_mean, fc1_stddev

    def inference2(self, latent2, class_labels, n_classes, is_training=True):
        with tf.variable_scope("inference2"):
            conv5 = conv2d_bn_lrelu(latent2, class_labels, n_classes, self.network.cs[3], [4, 4], 2, is_training, name='inference2/conv2d_1')
            conv6 = conv2d_bn_lrelu(conv5, class_labels, n_classes, self.network.cs[3], [4, 4], 1, is_training, name='inference2/conv2d_2')
            conv6 = tf.reshape(conv6, [-1, np.prod(conv6.get_shape().as_list()[1:])])
            return conv6

    def ladder2(self, latent2, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder2"):
            conv5 = conv2d_bn_lrelu(latent2, class_labels, n_classes, self.network.cs[3], [4, 4], 2, is_training, name='ladder2/conv2d_1')
            conv6 = conv2d_bn_lrelu(conv5, class_labels, n_classes, self.network.cs[3], [4, 4], 1, is_training, name='ladder2/conv2d_2')
            conv6 = tf.reshape(conv6, [-1, np.prod(conv6.get_shape().as_list()[1:])])
            fc2_mean = tf.contrib.layers.fully_connected(conv6, self.network.ladder2_dim, activation_fn=tf.identity)
            fc1_mean = scale_and_shift_flat(fc1_mean, class_labels, n_classes, name='ladder2/s_and_s_f_1')
            fc2_stddev = tf.contrib.layers.fully_connected(conv6, self.network.ladder2_dim, activation_fn=tf.sigmoid)
            fc1_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='ladder2/s_and_s_f_2')
            return fc2_mean, fc2_stddev

    def ladder3(self, latent3, class_labels, n_classes, is_training=True):
        with tf.variable_scope("ladder3"):
            fc1 = fc_bn_lrelu(latent3, class_labels, n_classes, self.network.cs[4], is_training, name='ladder3/fc_1')
            fc2 = fc_bn_lrelu(fc1, class_labels, n_classes, self.network.cs[4], is_training, name='ladder3/fc_2')
            fc3_mean = tf.contrib.layers.fully_connected(fc2, self.network.ladder3_dim, activation_fn=tf.identity)
            fc1_mean = scale_and_shift_flat(fc1_mean, class_labels, n_classes, name='ladder3/s_and_s_f_1')
            fc3_stddev = tf.contrib.layers.fully_connected(fc2, self.network.ladder3_dim, activation_fn=tf.sigmoid)
            fc1_stddev = scale_and_shift_flat(fc1_stddev, class_labels, n_classes, name='ladder3/s_and_s_f_2')
            return fc3_mean, fc3_stddev

    def generative0(self, latent1, ladder0, class_labels, n_classes, reuse=False, is_training=True):
        with tf.variable_scope("generative0") as gs:
            if reuse:
                gs.reuse_variables()
            if ladder0 is not None:
                ladder0 = fc_bn_relu(ladder0, class_labels, n_classes, int(self.network.fs[1] * self.network.fs[1] * self.network.cs[1]), is_training, name='generative0/s_and_s_f_1')
                ladder0 = tf.reshape(ladder0, [-1, self.network.fs[1], self.network.fs[1], self.network.cs[1]])
                if latent1 is not None:
                    latent1 = tf.concat(values=[latent1, ladder0], axis=3)
                else:
                    latent1 = ladder0
            elif latent1 is None:
                print("Generative layer must have input")
                exit(0)
            conv1 = conv2d_t_bn_relu(latent1, class_labels, n_classes, self.network.cs[1], [4, 4], 2, is_training, name='generative0/conv2d_1')
            output = tf.contrib.layers.convolution2d_transpose(conv1, self.network.data_dims[2], [4, 4], 1,
                                                               activation_fn=tf.sigmoid)
            output = scale_and_shift(output, class_labels, n_classes, name='generative0/s_and_s_f_2')
            output = (self.network.dataset.range[1] - self.network.dataset.range[0]) * output + self.network.dataset.range[0]
            return output

    def generative1(self, latent2, ladder1, class_labels, n_classes, reuse=False, is_training=True):
        with tf.variable_scope("generative1") as gs:
            if reuse:
                gs.reuse_variables()
            if ladder1 is not None:
                ladder1 = fc_bn_relu(ladder1, class_labels, n_classes, int(self.network.fs[2] * self.network.fs[2] * self.network.cs[2]), is_training, name='generative1/s_and_s_f_1')
                ladder1 = tf.reshape(ladder1, [-1, self.network.fs[2], self.network.fs[2], self.network.cs[2]])
                if latent2 is not None:
                    latent2 = tf.concat(values=[latent2, ladder1], axis=3)
                else:
                    latent2 = ladder1
            elif latent2 is None:
                print("Generative layer must have input")
                exit(0)
            conv1 = conv2d_t_bn_relu(latent2, class_labels, n_classes, self.network.cs[2], [4, 4], 2, is_training, name='generative1/conv2d_1')
            conv2 = conv2d_t_bn_relu(conv1, class_labels, n_classes, self.network.cs[1], [4, 4], 1, is_training, name='generative1/conv2d_2')
            return conv2

    def generative2(self, latent3, ladder2, class_labels, n_classes, reuse=False, is_training=True):
        with tf.variable_scope("generative2") as gs:
            if reuse:
                gs.reuse_variables()
            if ladder2 is not None:
                if latent3 is not None:
                    latent3 = tf.concat(values=[latent3, ladder2], axis=1)
                else:
                    latent3 = ladder2
            elif latent3 is None:
                print("Generative layer must have input")
                exit(0)
            fc1 = fc_bn_relu(latent3, class_labels, n_classes, int(self.network.fs[3] * self.network.fs[3] * self.network.cs[3]), is_training, name='generative2/fc_1')
            fc1 = tf.reshape(fc1, tf.stack([tf.shape(fc1)[0], self.network.fs[3], self.network.fs[3], self.network.cs[3]]))
            conv1 = conv2d_t_bn_relu(fc1, class_labels, n_classes, self.network.cs[3], [4, 4], 2, is_training, name='generative2/conv2d_1')
            conv2 = conv2d_t_bn_relu(conv1, class_labels, n_classes, self.network.cs[2], [4, 4], 1, is_training, name='generative2/conv2d_2')
            return conv2

    def generative3(self, latent4, ladder3, class_labels, n_classes, reuse=False, is_training=True):
        with tf.variable_scope("generative3") as gs:
            if reuse:
                gs.reuse_variables()
            fc1 = fc_bn_relu(ladder3, class_labels, n_classes, self.network.cs[4], is_training, name='generative3/fc_1')
            fc2 = fc_bn_relu(fc1, class_labels, n_classes, self.network.cs[4], is_training, name='generative3/fc_2')
            fc3 = fc_bn_relu(fc2, class_labels, n_classes, self.network.cs[4], is_training, name='generative3/fc_3')
            return fc3
