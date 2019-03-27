try:    # Works for python 3
    from dataset.dataset import *
    from dataset.dataset_atari import AtariDataset
except: # Works for python 2
    from dataset import *
    from dataset_atari import AtariDataset
