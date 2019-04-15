from visualize import *
import time
from glob import glob
from dataset.dataset_atari import AtariDataset
# Added in numpy.
import numpy as np


class NoisyTrainer:
    def __init__(self, network, dataset, args):
        self.network = network
        self.dataset = dataset
        self.args = args
        self.batch_size = args.batch_size
        self.data_dims = self.dataset.data_dims
        self.file_path = args.file_path
        if self.file_path[-1] != '/':
            self.file_path += '/'
        self.train_with_mask = False
        self.train_discrete = False

        self.fig, self.ax = None, None
        self.network = network
        self.test_reconstruction_error = True

    def get_noisy_input(self, original):
        if not self.args.denoise_train:
            return original

        noisy_input = np.multiply(original, np.random.binomial(n=1, p=0.9, size=[self.batch_size]+self.data_dims)) + \
                      np.random.binomial(n=1, p=0.1, size=[self.batch_size]+self.data_dims)

        noisy_input += np.random.normal(scale=0.1, size=[self.batch_size]+self.dataset.data_dims)

        return np.clip(noisy_input, a_min=self.dataset.range[0], a_max=self.dataset.range[1])

    def train(self):
        if self.network.do_generate_samples:
            sample_visualizer = SampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_conditional_samples:
            sample_visualizer_conditional = ConditionalSampleVisualizer(self.network, self.dataset)
        if self.network.do_generate_manifold_samples:
            sample_visualizer_manifold = ManifoldSampleVisualizer(self.network, self.dataset)

        iteration = 0
        while True:
            iter_time = time.time()
            images, classes = self.dataset.next_batch(self.batch_size)
            noisy_input = self.get_noisy_input(images)
            recon_loss, reg_loss, latent = self.network.train(noisy_input, images, classes)
            pth = self.file_path + "vladder_"+self.args.dataset+"_"+self.args.name

            if iteration % 20 == 0:
                print("Iteration %d: Reconstruction loss %f, Regularization loss %f, time per iter %fs" %
                      (iteration, recon_loss, reg_loss, time.time() - iter_time))
                if not os.path.exists(pth):
                    os.mkdir(pth)
                with open(pth+"/vladder_"+self.args.dataset+"_loss"+".txt", 'a') as f:
                    f.write("Iteration %d: Reconstruction loss %f, Regularization loss %f, time per iter %fs\n" %
                      (iteration, recon_loss, reg_loss, time.time() - iter_time))

            if iteration % self.args.vis_frequency == 0:
                test_error = self.test(iteration//self.args.vis_frequency, 5)
                print("Reconstruction error @%d per pixel: " % iteration, test_error)

                layers = [layer for layer in self.network.random_latent_code()]
                layers.sort()
                if self.network.do_generate_samples:
                    sample_visualizer.visualize(num_rows=10, use_gui=self.args.use_gui)

            iteration += 1

    def visualize(self):
        layers = [layer for layer in self.network.random_latent_code()]
        layers.sort()

        if self.network.do_generate_samples:
            sample_visualizer = SampleVisualizer(self.network, self.dataset)
            sample_visualizer.visualize(num_rows=10, use_gui=self.args.use_gui)

    """ Returns reconstruction error per pixel """
    def test(self, epoch, num_batch=3):
        error = 0.0
        for test_iter in range(num_batch):
            test_image, test_class = self.dataset.next_test_batch(self.batch_size)
            noisy_test_image = self.get_noisy_input(test_image)
            reconstruction = self.network.test(noisy_test_image, test_class)
            error += np.sum(np.square(reconstruction - test_image)) / np.prod(self.data_dims[:2]) / self.batch_size
            if test_iter == 0 and self.args.plot_reconstruction:
                self.plot_reconstruction(epoch, test_image, noisy_test_image, reconstruction)
        return error / num_batch

    def plot_reconstruction(self, epoch, test_image, noisy_image, reconstruction, num_plot=3):
        if test_image.shape[-1] == 1:
            canvas = np.zeros((num_plot*self.data_dims[0], 3*self.data_dims[1] + 20, self.data_dims[2]))
        else:
            canvas = np.ones((num_plot*self.data_dims[0], 3*self.data_dims[1] + 20, self.data_dims[2]))
        for img_index in range(num_plot):
            canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], 0:self.data_dims[1]] = \
                self.dataset.display(test_image[img_index, :, :])
            canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], self.data_dims[1]+10:self.data_dims[1]*2+10] = \
                self.dataset.display(noisy_image[img_index, :, :])
            canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], self.data_dims[1]*2+20:] = \
                self.dataset.display(reconstruction[img_index, :, :])

        img_folder = "models/" + self.network.name + "/reconstruction"
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        if canvas.shape[-1] == 1:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas[:, :, 0])
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas[:, :, 0])
        else:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas)
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas)

        if self.args.use_gui:
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.fig.suptitle("Reconstruction of " + str(self.network.name))
            self.ax.cla()
            if canvas.shape[-1] == 1:
                self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
            else:
                self.ax.imshow(canvas)
            plt.draw()
            plt.pause(1)
