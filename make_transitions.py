from glob import glob
from os import listdir, path
import numpy as np
import re

for folder in glob('grayscale/*'):
  for game in glob(folder + '/*'):
    pth = game + "/state_{}.npy"
    i = 0
    for filename in glob(game + '/*'):
      state = int(re.findall("\d+", filename)[-1])
      one = path.isfile(pth.format(state+1))
      two = path.isfile(pth.format(state+2))
      three = path.isfile(pth.format(state+3))
      if one and two and three:
        img1 = np.load(filename)
        img2 = np.load(pth.format(state+1))
        img3 = np.load(pth.format(state+2))
        img4 = np.load(pth.format(state+3))
        new_img = np.dstack((img1, img2, img3, img4))
        np.save("transitions/" + game[10:] + "/state_{}.npy".format(i), new_img)
        i += 1
