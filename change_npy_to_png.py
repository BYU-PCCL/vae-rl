import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob

fpath = sys.argv[1]

for f in glob(fpath+"*.npy"):
  if not os.path.isfile(f[:-4]+"_0.png"):
    img = np.load(f)
    for i in range(4):
      plt.imshow(img[:,:,i], cmap="gray")
      plt.savefig(f[:-4]+"_{}.png".format(i))
    os.remove(f)
