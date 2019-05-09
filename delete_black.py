import numpy as np
from glob import glob
from os import remove

for folder in ['every_timestep/', 'grayscale/']:
  for game in glob(folder+'*/'):
    for i in range(10):
      for file_ in glob(game+'game_{}/state_*.npy'.format(i)):
        img = np.load(file_)
        if np.min(img) == 0.0 and np.max(img) == 0.0:
          remove(file_)
        

