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

class AtariDataset(Dataset):
    def __init__(self, transition=False, db_path = '', crop = True):
        self.transition = transition
        if transition:
          db_path = "../../../../not_backed_up/atarigames/transitions/"
        else:
          db_path = "../../../../not_backed_up/atarigames/all_games_uneven/"
        Dataset.__init__(self)
        self.data_files = []
        for folder in ['air_raid', 'atlantis', 'gravitar', 'name_this_game', 'river_raid', 
                       'sea_quest', 'solaris', 'space_invaders', 'time_pilot', 'zaxxon']:
            game_files = []
            for i in range(1, 11):
                for filename in glob(db_path + folder + "/game_{}/*.npy".format(i)):
                    game_files.append(filename)
            game_files = random.sample(game_files, 7400)
            self.data_files.extend(game_files)
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
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images


    def next_test_batch(self, batch_size):
        np.random.shuffle(self.test_img)
        sample_files = self.test_img[:batch_size]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def batch_by_index(self, batch_start, batch_end):
        sample_files = self.data_files[batch_start:batch_end]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    @staticmethod
    def downsample_image(image):
        red_image = block_reduce(image, block_size=(3, 2, 1), func=np.max)
        return red_image

    @staticmethod
    def get_image(image_path):
        image = np.load(image_path)
        image = AtariDataset.downsample_image(image)
        temp_img = image.copy()
        x, y, z = image.shape
        x, y = np.float(96 - x), np.float(96 - y)
        if x % 2 == 0:
             x1, x2 = int(x/2), int(x/2)
        else:
             x1, x2 = int(np.floor(x/2)), int(np.ceil(x/2))
        if y % 2 == 0:
             y1, y2 = int(y/2), int(y/2)
        else:
             y1, y2 = int(np.floor(y/2)), int(np.ceil(y/2))
        image = np.pad(image, ((x1,x2),(y1,y2),(0,0)),'edge')
        image = image.astype('float32')  
        image = image * 2 - 1.0
#        image = image / 127.5 - 1.0
        return image

    @staticmethod
    def center_crop(x, crop_h, crop_w=None, resize_w=96):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        return misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_w, resize_w])

    @staticmethod
    def full_crop(x):
        if x.shape[0] <= x.shape[1]:
            lb = int((x.shape[1] - x.shape[0]) / 2)
            ub = lb + x.shape[0]
            x = misc.imresize(x[:, lb:ub], [96, 96])
        else:
            lb = int((x.shape[0] - x.shape[1]) / 2)
            ub = lb + x.shape[1]
            x = misc.imresize(x[lb:ub, :], [96, 96])
        return x

    @staticmethod
    def transform(image, npx=108, is_crop=True, resize_w=96):
        # npx : # of pixels width/height of image
        if is_crop:
            cropped_image = AtariDataset.center_crop(image, npx, resize_w=resize_w)
        else:
            cropped_image = AtariDataset.full_crop(image)
        return np.array(cropped_image) / 127.5 - 1.

    """ Transform image to displayable """
    def display(self, image):
        rescaled = np.divide(image + 1.0, 2.0)
        return np.clip(rescaled, 0.0, 1.0)

    def reset(self):
        self.idx = 0

