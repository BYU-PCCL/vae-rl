from dataset import *
import math, os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.misc as misc
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
import random
import cv2

class AtariDataset(Dataset):
    def __init__(self, transition=False, db_path = '', crop = True):
        self.transition = transition
        if transition:
          db_path = "../../../../not_backed_up/atarigames/transitions/"
        else:
          db_path = "../../../../not_backed_up/atarigames/every_timestep/"
        Dataset.__init__(self)
        self.data_files = []
        j = 1
        for folder in ['air_raid', 'atlantis', 'gravitar', 'name_this_game', 'river_raid', 
                       'sea_quest', 'solaris', 'space_invaders', 'time_pilot', 'zaxxon']:
            game_files = []
            for i in range(1, 11):
                for filename in glob(db_path + folder + "/game_{}/*.npy".format(i)):
                    game_files.append([filename, j])
#                    game_files.append(filename)
            if self.transition:
                game_files = random.sample(game_files, 7400)
            else:
                game_files = random.sample(game_files, 5640)
            self.data_files.extend(game_files)
            j += 1
        np.random.shuffle(self.data_files)
        self.train_size = int(float(len(self.data_files) * 0.8))
        self.test_size = len(self.data_files) - self.train_size
        self.train_img = self.data_files[:self.train_size]
        self.test_img = self.data_files[self.train_size:]

        self.train_idx = 0
        self.test_idx = 0
        size = 96
        if transition:
            self.channels = 4
        else:
            self.channels = 3

        self.data_dims = [size, size, self.channels]
        self.train_cache = np.ndarray((self.train_size, size, size, self.channels), dtype=np.float32)
        self.train_cache_top = 0
        self.test_cache = np.ndarray((self.test_size, size, size, self.channels), dtype=np.float32)
        self.test_cache_top = 0
        self.range = [-1., 1.]
        self.is_crop = crop
        self.name = "atari"

    def next_batch(self, batch_size):
        np.random.shuffle(self.train_img)
        sample_files = self.train_img[:batch_size]
        sample = np.array([self.get_image(sample_file) for sample_file in sample_files])
        sample_images = np.stack(sample[:, 0]).astype(np.float32)
        sample_classes = np.stack(sample[:, 1]).astype(np.float32)
        return sample_images, sample_classes

    def next_test_batch(self, batch_size):
        np.random.shuffle(self.test_img)
        sample_files = self.test_img[:batch_size]
        sample = np.array([self.get_image(sample_file) for sample_file in sample_files])
        sample_images = np.stack(sample[:, 0]).astype(np.float32)
        sample_classes = np.stack(sample[:, 1]).astype(np.float32)
        return sample_images, sample_classes

    def batch_by_index(self, batch_start, batch_end):
        print("ENTERED IN BATCH_BY_INDEX SO YOU DON'T NEED TO DELETE IT")
        sample_files = self.data_files[batch_start:batch_end]
        sample = np.array([self.get_image(sample_file) for sample_file in sample_files])
        sample_images = sample[:, 0]
        sample_classes = sample[:, 1]
        return sample_images, sample_classes

    def downsample_image(self, image):
        red_image = block_reduce(image, block_size=(3, 2, 1), func=np.max)
        return red_image

    def get_image(self, sample_file):
        image_path = sample_file[0]
        c = sample_file[1]
        image = np.load(image_path)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_NEAREST)
        image = image.astype('float32')
        if self.transition:
             image = image * 2 - 1.0
        else:
             image = image / 127.5 - 1.0
        one_hot = np.zeros(10)
        one_hot[c-1] = 1
        return image, one_hot

    """ Transform image to displayable """
    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)

    def reset(self):
        self.idx = 0


"""
    def next_batch(self, batch_size):
        np.random.shuffle(self.train_img)
        sample_files = self.train_img[:batch_size]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        print(np.shape(sample))
        sample_images = np.array(sample).astype(np.float32)
        return sample_images
    def get_image(self, image_path):
        image = np.load(image_path)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_NEAREST)
        image = image.astype('float32')
        if self.transition:
             image = image * 2 - 1.0
        else:
             image = image / 127.5 - 1.0
        return image
    def next_test_batch(self, batch_size):
        np.random.shuffle(self.test_img)
        sample_files = self.test_img[:batch_size]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images
"""
