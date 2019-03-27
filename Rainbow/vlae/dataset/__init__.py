try:    # Works for python 3
    from vlae.dataset import *
    from vlae.dataset.dataset_atari import AtariDataset
except: # Works for python 2
    from vlae.dataset.dataset import *
    from vlae.dataset.dataset_atari import AtariDataset
