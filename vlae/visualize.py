from abstract_network import *
import os
import scipy.misc as misc
# Imported numpy.
import numpy as np

class Visualizer:
    def __init__(self, network):
        plt.ion()
        plt.show()
        self.fig = None
        self.network = network
        self.name = "default"
        self.save_epoch = 0
        self.array_save_epoch = 0

    def visualize(self, **args):
        pass

    def fig_to_file(self, fig=None):
        if fig is None:
            fig = self.fig
        img_folder = self.network.file_path + "images/" + self.name
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        fig_name = "current.png"
        fig.savefig(os.path.join(img_folder, fig_name))
        fig_name = "epoch%d" % self.save_epoch + ".png"
        fig.savefig(os.path.join(img_folder, fig_name))
        self.save_epoch += 1

    def arr_to_file(self, arr, trans=False):
        img_folder = self.network.file_path + self.network.name + "_reconstructions/" + self.name
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        if arr.shape[-1] == 1:
            if not trans:
                misc.imsave(os.path.join(img_folder, 'current.png'), arr[:, :, 0])
                misc.imsave(os.path.join(img_folder, 'epoch%d.png' % self.save_epoch), arr[:, :, 0])
            else:
                np.save(img_folder + "/epoch{}.npy".format(self.save_epoch), arr[:, :, 0])
        else:
            if not trans:
                misc.imsave(os.path.join(img_folder, 'current.png'), arr)
                misc.imsave(os.path.join(img_folder, 'epoch%d.png' % self.save_epoch), arr)
            else:
                np.save(img_folder + "/epoch{}.npy".format(self.save_epoch), arr)
        self.save_epoch += 1

class SampleVisualizer(Visualizer):
    def __init__(self, network, dataset, trans=False):
        Visualizer.__init__(self, network)
        self.dataset = dataset
        self.name = "samples"
        self.trans = trans

    def visualize(self, num_rows=10, use_gui=False):
        if use_gui and self.fig is None:
            self.fig, self.ax = plt.subplots()
        samples = self.network.generate_samples()
        if samples is not None:
            samples = self.dataset.display(samples)
            width = samples.shape[1]
            height = samples.shape[2]
            channel = samples.shape[3]
            canvas = np.zeros((width * num_rows, height * num_rows, channel))
            for img_index1 in range(num_rows):
                for img_index2 in range(num_rows):
                    canvas[img_index1*width:(img_index1+1)*width, img_index2*height:(img_index2+1)*height, :] = \
                        samples[img_index1*num_rows+img_index2, :, :, :]
            self.arr_to_file(canvas, self.trans)

            if use_gui:
                self.ax.cla()
                if channel == 1:
                    self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
                else:
                    self.ax.imshow(canvas)
                self.ax.xaxis.set_visible(False)
                self.ax.yaxis.set_visible(False)

                self.fig.suptitle('Samples for %s' % self.network.name)
                plt.draw()
                plt.pause(0.01)
