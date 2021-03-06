from vlae.vladder_large import *
from vlae.vladder_medium import *
from vlae.vladder_small import *
import numpy as np


class VLadder(Network):
    def __init__(self, dataset, file_path, name=None, reg='kl', batch_size=100, restart=False, add_coords=True):
        Network.__init__(self, dataset, batch_size, file_path)
        self.name = "vladder_atari_{}".format(name)
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dims = self.dataset.data_dims
        self.latent_noise = False
        self.file_path = file_path

        self.fs = [self.data_dims[0], self.data_dims[0] // 2, self.data_dims[0] // 4, self.data_dims[0] // 8,
                   self.data_dims[0] // 16]
        self.reg = reg
        if self.reg != 'kl' and self.reg != 'mmd':
            print("Unknown regularization, supported: kl, mmd")

        self.cs = [3, 64, 128, 256, 512, 1024]
        self.ladder0_dim = 21
        self.ladder1_dim = 21
        self.ladder2_dim = 21
        self.ladder3_dim = 21
        self.num_layers = 4
        loss_ratio = 0.5
        layers = LargeLayers(self, add_coords)
        self.do_generate_conditional_samples = False
        self.do_generate_samples = True

        self.self = self

        self.input_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="input_placeholder")
        self.target_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="target_placeholder")
        self.classes_placeholder = tf.placeholder(shape=[None]+[10], dtype = tf.float32, name="classes_placeholder")
        self.is_training = tf.placeholder(tf.bool, name='phase')
        
        self.regularization = 0.0
        input_size = tf.shape(self.input_placeholder)[0]
        if self.ladder0_dim > 0:
            self.iladder0_mean, self.iladder0_stddev = layers.ladder0(self.input_placeholder, self.classes_placeholder, is_training=self.is_training)
            self.iladder0_stddev += 0.001
            self.iladder0_sample = self.iladder0_mean + \
                tf.multiply(self.iladder0_stddev, tf.random_normal(tf.stack([input_size, self.ladder0_dim])))
            if self.reg == 'kl':
                self.ladder0_reg = -tf.log(self.iladder0_stddev) + 0.5 * tf.square(self.iladder0_stddev) + \
                                    0.5 * tf.square(self.iladder0_mean) - 0.5
                self.ladder0_reg = tf.reduce_mean(tf.reduce_sum(self.ladder0_reg, axis=1))
            elif self.reg == 'mmd':
                prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder0_dim])
                self.ladder0_reg = compute_mmd(self.iladder0_sample, prior_sample)
            tf.summary.scalar("ladder0_reg", self.ladder0_reg)
            self.regularization += self.ladder0_reg

        if self.num_layers >= 2:
            self.ilatent1_hidden = layers.inference0(self.input_placeholder, self.classes_placeholder, is_training=self.is_training)
            if self.ladder1_dim > 0:
                self.iladder1_mean, self.iladder1_stddev = layers.ladder1(self.ilatent1_hidden, self.classes_placeholder, is_training=self.is_training)
                self.iladder1_stddev += 0.001
                self.iladder1_sample = self.iladder1_mean + \
                    tf.multiply(self.iladder1_stddev, tf.random_normal(tf.stack([input_size, self.ladder1_dim])))

                if self.reg == 'kl':
                    self.ladder1_reg = -tf.log(self.iladder1_stddev) + 0.5 * tf.square(self.iladder1_stddev) + \
                                        0.5 * tf.square(self.iladder1_mean) - 0.5
                    self.ladder1_reg = tf.reduce_mean(tf.reduce_sum(self.ladder1_reg, axis=1))
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder1_dim])
                    self.ladder1_reg = compute_mmd(self.iladder1_sample, prior_sample)
                tf.summary.scalar("ladder1_reg", self.ladder1_reg)
                self.regularization += self.ladder1_reg

        if self.num_layers >= 3:
            self.ilatent2_hidden = layers.inference1(self.ilatent1_hidden, self.classes_placeholder, is_training=self.is_training)
            if self.ladder2_dim > 0:
                self.iladder2_mean, self.iladder2_stddev = layers.ladder2(self.ilatent2_hidden, self.classes_placeholder, is_training=self.is_training)
                self.iladder2_stddev += 0.001
                self.iladder2_sample = self.iladder2_mean + \
                    tf.multiply(self.iladder2_stddev, tf.random_normal(tf.stack([input_size, self.ladder2_dim])))

                if self.reg == 'kl':
                    self.ladder2_reg = -tf.log(self.iladder2_stddev) + 0.5 * tf.square(self.iladder2_stddev) + \
                                        0.5 * tf.square(self.iladder2_mean) - 0.5
                    self.ladder2_reg = tf.reduce_mean(tf.reduce_sum(self.ladder2_reg, axis=1))
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder2_dim])
                    self.ladder2_reg = compute_mmd(self.iladder2_sample, prior_sample)
                tf.summary.scalar("latent2_kl", self.ladder2_reg)
                self.regularization += self.ladder2_reg

        if self.num_layers >= 4:
            self.ilatent3_hidden = layers.inference2(self.ilatent2_hidden, self.classes_placeholder, is_training=self.is_training)
            if self.ladder3_dim > 0:
                self.iladder3_mean, self.iladder3_stddev = layers.ladder3(self.ilatent3_hidden, self.classes_placeholder, is_training=self.is_training)
                self.iladder3_stddev += 0.001
                self.iladder3_sample = self.iladder3_mean + \
                    tf.multiply(self.iladder3_stddev, tf.random_normal(tf.stack([input_size, self.ladder3_dim])))

                if self.reg == 'kl':
                    self.ladder3_reg = -tf.log(self.iladder3_stddev) + 0.5 * tf.square(self.iladder3_stddev) + \
                                        0.5 * tf.square(self.iladder3_mean) - 0.5
                    self.ladder3_reg = tf.reduce_mean(tf.reduce_sum(self.ladder3_reg, axis=1))
                elif self.reg == 'mmd':
                    prior_sample = tf.random_normal(shape=[self.batch_size, self.ladder3_dim])
                    self.ladder3_reg = compute_mmd(self.iladder3_sample, prior_sample)
                tf.summary.scalar("latent3_kl", self.ladder3_reg)
                self.regularization += self.ladder3_reg

        self.latent = tf.concat([self.iladder0_sample, self.iladder1_sample, self.iladder2_sample, self.iladder3_sample], 1)

        self.ladders = {}
        if self.num_layers >= 4 and self.ladder3_dim > 0:
            self.ladder3_placeholder = tf.placeholder(shape=(None, self.ladder3_dim), dtype=tf.float32, name="ladder3")
            self.ladders['ladder3'] = [self.ladder3_placeholder, self.ladder3_dim, self.iladder3_sample]
            tlatent3_state = layers.generative3(None, self.iladder3_sample, self.classes_placeholder, is_training=self.is_training)
            glatent3_state = layers.generative3(None, self.ladder3_placeholder, self.classes_placeholder, reuse=True, is_training=False)
        else:
            tlatent3_state, glatent3_state = None, None

        if self.num_layers >= 3 and self.ladder2_dim > 0:
            self.ladder2_placeholder = tf.placeholder(shape=(None, self.ladder2_dim), dtype=tf.float32, name="ladder2")
            self.ladders['ladder2'] = [self.ladder2_placeholder, self.ladder2_dim, self.iladder2_sample]
            tlatent2_state = layers.generative2(tlatent3_state, self.iladder2_sample, self.classes_placeholder, is_training=self.is_training)
            glatent2_state = layers.generative2(glatent3_state, self.ladder2_placeholder, self.classes_placeholder, reuse=True, is_training=False)
        elif tlatent3_state is not None:
            tlatent2_state = layers.generative2(tlatent3_state, None, self.classes_placeholder, is_training=self.is_training)
            glatent2_state = layers.generative2(glatent3_state, None, self.classes_placeholder, reuse=True, is_training=False)
        else:
            tlatent2_state, glatent2_state = None, None

        if self.num_layers >= 2 and self.ladder1_dim > 0:
            self.ladder1_placeholder = tf.placeholder(shape=(None, self.ladder1_dim), dtype=tf.float32, name="ladder1")
            self.ladders['ladder1'] = [self.ladder1_placeholder, self.ladder1_dim, self.iladder1_sample]
            tlatent1_state = layers.generative1(tlatent2_state, self.iladder1_sample, self.classes_placeholder, is_training=self.is_training)
            glatent1_state = layers.generative1(glatent2_state, self.ladder1_placeholder, self.classes_placeholder, reuse=True, is_training=False)
        elif tlatent2_state is not None:
            tlatent1_state = layers.generative1(tlatent2_state, None, self.classes_placeholder, is_training=self.is_training)
            glatent1_state = layers.generative1(glatent2_state, None, self.classes_placeholder, reuse=True, is_training=False)
        else:
            tlatent1_state, glatent1_state = None, None

        if self.ladder0_dim > 0:
            self.ladder0_placeholder = tf.placeholder(shape=(None, self.ladder0_dim), dtype=tf.float32, name="ladder0")
            self.ladders['ladder0'] = [self.ladder0_placeholder, self.ladder0_dim, self.iladder0_sample]
            self.toutput = layers.generative0(tlatent1_state, self.iladder0_sample, self.classes_placeholder, is_training=self.is_training)
            self.goutput = layers.generative0(glatent1_state, self.ladder0_placeholder, self.classes_placeholder, reuse=True, is_training=False)
        elif tlatent1_state is not None:
            self.toutput = layers.generative0(tlatent1_state, None, self.classes_placeholder, is_training=self.is_training)
            self.goutput = layers.generative0(glatent1_state, None, self.classes_placeholder, reuse=True, is_training=False)
        else:
            print("Error: no active ladder")
            exit(0)

        self.reconstruction_loss = tf.reduce_mean(tf.abs(self.toutput - self.target_placeholder))

        self.reg_coeff = tf.placeholder_with_default(1.0, shape=[], name="regularization_coeff")

        if self.reg == 'kl':
            self.reconstruction_loss *= loss_ratio * np.prod(self.data_dims)
            self.loss = self.reg_coeff * self.regularization + self.reconstruction_loss        
        elif self.reg == 'mmd':
            self.regularization *= 100
            self.reconstruction_loss *= 100
            self.loss = self.regularization + self.reconstruction_loss

        tf.summary.scalar("reconstruction_loss", self.reconstruction_loss)
        tf.summary.scalar("regularization_loss", self.regularization)
        tf.summary.scalar("loss", self.loss)

        self.merged_summary = tf.summary.merge_all()
        self.iteration = 0

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(0.0002).minimize(self.loss)
        self.init_network(restart=restart)
        self.print_network()
        self.read_only = False

    def train(self, batch_input, batch_target, batch_classes, label=None):
        self.iteration += 1

        codes = {key: np.random.normal(size=[self.batch_size, self.ladders[key][1]]) for key in self.ladders}
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict.update({
            self.input_placeholder: batch_input,
            self.reg_coeff: 1 - math.exp(-self.iteration / 2000.0),
            self.target_placeholder: batch_target,
            self.classes_placeholder: batch_classes,
            self.is_training: True
        })
        _, recon_loss, reg_loss = self.sess.run([self.train_op, self.reconstruction_loss, self.regularization],
                                                feed_dict=feed_dict)
        if self.iteration % 2000 == 0:
            self.save_network()
        if self.iteration % 20 == 0:
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.iteration)
        return recon_loss, reg_loss, self.sess.run(self.latent, feed_dict=feed_dict) 

    def test(self, batch_input, batch_classes, label=None):
        train_return = self.sess.run(self.toutput,
                                     feed_dict={self.input_placeholder: batch_input,
                                                self.classes_placeholder: batch_classes,
                                                self.is_training: False})
        return train_return

    def random_latent_code(self):
        return {key: np.random.normal(size=[self.ladders[key][1]]) for key in self.ladders}

    def generate_samples(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        codes = {key: np.random.normal(size=[batch_size, self.ladders[key][1]]) for key in self.ladders}
        batch_classes = np.eye(10)[np.random.choice(10, batch_size)]
        feed_dict = {self.ladders[key][0]: codes[key] for key in self.ladders}
        feed_dict[self.is_training] = False
        feed_dict[self.classes_placeholder] = batch_classes
        output = self.sess.run(self.goutput, feed_dict=feed_dict)
        return output

    def get_latent_codes(self, image, trans=False):
        image = tf.cast(image, tf.float32)
        if trans:
            image = tf.scalar_mul(2, image)
        else:
            image = tf.divide(image, 127.5)
        image = tf.subtract(image, 1.0)
        c = np.zeros(10)
        c[5] = 1
        feed_dict = {self.input_placeholder: image.eval(),
                     self.classes_placeholder: np.reshape(c,(1,10)),
                     self.is_training: False}
        return self.sess.run(self.latent, feed_dict=feed_dict)
