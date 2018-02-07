
# coding: utf-8


import tensorflow as tf
import numpy as np
import os
import random
from matplotlib import pyplot as plt


# ### Define Hyperparameters


batch_size = 16
n_classes = 4


# ### Define Graph


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def scale_and_shift(x, labels, reuse=True):
    with tf.variable_scope("scale_and_shift", reuse=reuse):
        axis = [1,2]
        x_shape = x.get_shape().as_list()
        print('x shape: {}'.format(x_shape))
        beta = tf.get_variable( 'beta', [n_classes])# ,
                                     # initializer=tf.zeros_initializer())
        print('beta shape: {}'.format(beta.get_shape().as_list()))
        gamma = tf.get_variable( 'gamma', [n_classes])#,
                                    # initializer=tf.ones_initializer())
        print('gamma shape: {gamma.get_shape().as_list()}')
        
        conditioned_shift = tf.gather(beta, labels)
        print('conditioned_shift shape: {}'.format(conditioned_shift.get_shape().as_list()))
        conditioned_shift = tf.expand_dims(tf.expand_dims(conditioned_shift, 1), 1)
        print('conditioned_shift shape: {}'.format(conditioned_shift.get_shape().as_list()))
        
        conditioned_scale = tf.gather(gamma, labels)
        print('conditioned_scale shape: {}'.format(conditioned_scale.get_shape().as_list()))
        conditioned_scale = tf.expand_dims(tf.expand_dims(conditioned_scale, 1), 1)
        print('conditioned_scale shape: {}'.format(conditioned_scale.get_shape().as_list()))
        
        mean, variance = tf.nn.moments(x, axis, keep_dims=True)
        print('mean shape: {}'.format(mean.get_shape().as_list()))
        print('variance shape: {}'.format(variance.get_shape().as_list()))
        
        moments_shape = tf.shape(mean)
        mu = tf.zeros(moments_shape)
        sigma = tf.ones(moments_shape)
        print('mu shape: {}'.format(mu.get_shape().as_list()))
        print('sigma shape: {}'.format(sigma.get_shape().as_list()))
        
        variance_epsilon = 0.01
#         output = tf.nn.batch_normalization(x=x, mean=conditioned_shift,
#                                            variance=conditioned_scale,
#                                            offset=None, scale=None,
#                                            variance_epsilon=variance_epsilon)
        output = tf.nn.batch_normalization(x=x, mean=mu,
                                           variance=sigma,
                                           offset=conditioned_shift, scale=conditioned_scale,
                                           variance_epsilon=variance_epsilon)
        print('output shape: {}'.format(output.get_shape().as_list()))
        return output

    

def encoder(X_in, labels, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 210, 160, 3])
        
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        print('x shape: {}'.format(tf.shape(x)))
        x = scale_and_shift(x, labels, reuse=False)
        x = tf.nn.dropout(x, keep_prob)
        
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = scale_and_shift(x, labels, reuse=True)
        x = tf.nn.dropout(x, keep_prob)
        
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        
        x = tf.contrib.layers.flatten(x)
        print('final x: {}'.format(x.get_shape().as_list()))
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=(inputs_decoder * 2 + 1), activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=210*160*3, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 210, 160, 3])
        return img


# ### Initialize Graph


tf.reset_default_graph()

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 210, 160, 3], name='X')
Labels = tf.placeholder(dtype=tf.int32, shape=[None], name='Labels')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 210, 160, 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 210*160*3])
print('Y_flat: {}'.format(Y_flat.get_shape().as_list()))
print('Labels: {}'.format(Labels.get_shape().as_list()))
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels // 2

sampled, mn, sd = encoder(X_in, Labels, keep_prob)
dec = decoder(sampled, keep_prob)
print('dec: {}'.format(dec.get_shape().as_list()))

unreshaped = tf.reshape(dec, [-1, 210*160*3])
print('unreshaped: {}'.format(unreshaped.get_shape().as_list()))
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
print('here0')
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
print('here1')
sess = tf.Session()
print('here2')
sess.run(tf.global_variables_initializer())


# ### Gather New Training Data

directories = ['jamesbond', 'spaceinvaders', 'tutankham', 'zaxxon']
state_label_pairs = []
for i, root_dir in enumerate(directories):
    for dir_name, subdir_list, file_list in os.walk(root_dir+'/'):
        for fname in file_list:
            state_label_pairs.append((root_dir + '/' + fname, i))

print('Found {} files.'.format(len(state_list)))


# ### Train Model

def read_image(filename):
    image = np.load(filename)
    print(image.shape)
    plt.imshow(image)
    plt.show()

for i in range(400):
    next_batch = random.sample(state_label_pairs, batch_size)
    batch = [read_image(b[0]) for b in next_batch]
    labels = [lab[1] for lab in next_batch]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, Labels: labels, keep_prob: 0.8})
        
    if not i % 200:
        ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(read_image(next_batch[0][0]), [210, 160, 3]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls))


# ### Sample New Images

n_samples = 10
randoms = [np.random.normal(0, 1, n_latent) for _ in range(n_samples)]
classes = [np.random.choice(n_classes) for _ in range(n_samples)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [210, 160, 3]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img, cmap='gray')

