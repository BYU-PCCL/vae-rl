from dataset import *
import math, os
from glob import glob
import numpy as np
import scipy.misc as misc
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

class AtariDataset(Dataset):
    def __init__(self, crop=True):
        db_path = "../../../not_backed_up/atarigames/all_games_uneven/" # + db_path
        Dataset.__init__(self)
        self.data_files = []

        # db_path = "../../../not_backed_up/atarigames/shooting_even/"
        # Dataset.__init__(self)
        # self.data_files = []
        self.class_labels = []
        
        cur_label = 0
        for folder in glob(db_path+"*/"):
             for game in glob(folder+"*/"):
                  for i in range(1, 11):
                       for file_ in glob(game+"game_{}/".format(i)+"*.npy"):
                            self.data_files.append(file_)
                            self.class_labels.append(cur_label)
                  cur_label += 1

        # for label, folder in enumerate(glob(db_path + "*/")):
        #     # self.class_labels.append(label)
        #     for i in range(1, 11):
        #         for file_ in glob(folder + "game_{}/".format(i) + "*.npy"):
        #             self.data_files.append(file_)
        #             self.class_labels.append(label)
        
        self.n_classes = max(self.class_labels) + 1
        print('Number of classes: {}'.format(self.n_classes))
        print('Number of class labels: {}'.format(len(self.class_labels))) 
        
        # shuffle data_files and class labels together
        data_files_labels = list(zip(self.data_files, self.class_labels))
        # print(data_files_labels)
        
        np.random.shuffle(data_files_labels)
        self.data_files, self.class_labels = zip(*data_files_labels)

        self.train_size = int(float(len(self.data_files) * 0.8))
        self.test_size = len(self.data_files) - self.train_size
        
        self.train_img = self.data_files[:self.train_size]
        self.train_class_labels = self.class_labels[:self.train_size]

        self.test_img = self.data_files[self.train_size:]
        self.test_class_labels = self.class_labels[self.train_size:]

        self.train_idx = 0
        self.test_idx = 0
        self.data_dims = [96, 96, 3]

        self.train_cache = np.ndarray((self.train_size, 96, 96, 3), dtype=np.float32)
        self.train_label_cache = np.ndarray((self.train_size), dtype=np.int32)
        self.train_cache_top = 0
        self.test_cache = np.ndarray((self.test_size, 96, 96, 3), dtype=np.float32)
        self.test_label_cache = np.ndarray((self.test_size), dtype=np.int32)
        self.test_cache_top = 0
        self.range = [-1., 1.]
        self.is_crop = crop
        self.name = "atari"

    def next_batch(self, batch_size):
        all_data = list(zip(self.train_img, self.train_class_labels))
        np.random.shuffle(all_data)
        self.train_img, self.train_class_labels = zip(*all_data)

        sample_files = self.train_img[:batch_size]
        sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        sample_class_labels = self.train_class_labels[:batch_size]
        return sample_images, sample_class_labels

    def next_test_batch(self, batch_size):
        c = list(zip(self.test_img, self.test_class_labels))
        np.random.shuffle(c)
        self.test_img, self.test_class_labels = zip(*c)

        sample_files = self.test_img[:batch_size]
        sample = [self.get_image(sample_file, self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        sample_class_labels = self.test_class_labels[:batch_size]
        return sample_images, sample_class_labels

    def batch_by_index(self, batch_start, batch_end):
        sample_files = self.data_files[batch_start:batch_end]
        sample = [self.get_image(sample_file) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        sample_class_labels = self.class_labels[batch_start:batch_end]
        return sample_images, sample_class_labels

    @staticmethod
    def downsample_image(image):
        red_image = block_reduce(image, block_size=(3, 2, 1), func=np.max)
        return red_image

    @staticmethod
    def get_image(image_path, is_crop):
        image = np.load(image_path)
        image = AtariDataset.downsample_image(image)
        temp_img = image.copy()
        x, y, z = image.shape
        x, y = 96 - x, 96 - y
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
        image = image / 127.5 - 1.0
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
