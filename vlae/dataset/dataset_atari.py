from dataset import *
import math, os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.misc as misc
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

class AtariDataset(Dataset):
    def __init__(self, db_path = '', crop = True):
        db_path = "../../../not_backed_up/atarigames/" + db_path
        Dataset.__init__(self)
        self.data_files = []
        if(len(glob(db_path+"*/"+"game_1/"))!=0):
            for folder in glob(db_path + "*/"):
                print(folder)
                for i in range(1, 11):
                    for file_ in glob(folder + "game_{}/".format(i) + "*.npy"):
                        self.data_files.append(file_)
        else:
            for i in range(1, 11):
                for file_ in glob(db_path + "game_{}/".format(i) + "*.npy"):
                    self.data_files.append(file_)
        np.random.shuffle(self.data_files)
        self.train_size = int(float(len(self.data_files) * 0.8))
        self.test_size = len(self.data_files) - self.train_size
        self.train_img = self.data_files[:self.train_size]
        self.test_img = self.data_files[self.train_size:]

        self.train_idx = 0
        self.test_idx = 0
        size = 96
        self.data_dims = [size, size, 3]

        self.train_cache = np.ndarray((self.train_size, size, size, 3), dtype=np.float32)
        self.train_cache_top = 0
        self.test_cache = np.ndarray((self.test_size, size, size, 3), dtype=np.float32)
        self.test_cache_top = 0
        self.range = [-1., 1.]
        self.is_crop = crop
        self.name = "atari"

    def next_batch(self, batch_size):
        # sample_files = self.data[0:batch_size]
        np.random.shuffle(self.train_img)
        """prev_idx = self.train_idx
        self.train_idx += batch_size
        if self.train_idx > self.train_size:
            self.train_idx = batch_size
            prev_idx = 0

        if self.train_idx < self.train_cache_top:
            return self.train_cache[prev_idx:self.train_idx, :, :, :]
        else:
            sample_files = self.train_img[prev_idx:self.train_idx]
            sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.train_cache[prev_idx:self.train_idx] = sample_images
            self.train_cache_top = self.train_idx
            return sample_images"""
        sample_files = self.train_img[:batch_size]
        sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images


    def next_test_batch(self, batch_size):
        """prev_idx = self.test_idx
        self.test_idx += batch_size
        if self.test_idx > self.test_size:
            self.test_idx = batch_size
            prev_idx = 0

        if self.test_idx < self.test_cache_top:
            return self.test_cache[prev_idx:self.test_idx, :, :, :]
        else:
            sample_files = self.test_img[prev_idx:self.test_idx]
            sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            self.test_cache[prev_idx:self.test_idx] = sample_images
            self.test_cache_top = self.test_idx
            return sample_images"""
        np.random.shuffle(self.test_img)
        sample_files = self.test_img[:batch_size]
        sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
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
    def get_image(image_path, is_crop):
        image = np.load(image_path)
        image = AtariDataset.downsample_image(image)
        x, y, z = image.shape
        x = int((96-x)/2)
        y = int((96-y)/2)
        image = np.pad(image, ((x,x),(y,y),(0,0)),'edge')
	#image = AtariDataset.transform(image, is_crop = is_crop)
        return image

    @staticmethod
    def center_crop(x, crop_h, crop_w=None, resize_w=96):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        print(h, w, j, i)
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


if __name__ == '__main__':
    dataset = AtariDataset()
    if not os.path.exists("crop_pad/"):
        os.mkdir("crop_pad/")
    i = 0
    while True:
        batch = dataset.next_batch(64)
        print(batch.shape)
        plt.imshow(dataset.display(batch[0]))
        i += 1
        plt.savefig("crop_pad/batch_{}.png".format(i))
        plt.show()
