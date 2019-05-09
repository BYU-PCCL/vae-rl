import numpy as np
from glob import glob
from skimage.color import rgb2gray
import re

for game in glob('every_timestep/*/'):
  print(game)
  for i in range(10):
    for file_ in glob(game + 'game_{}/*.npy'.format(i+1)):
      rgb_img = np.load(file_)
      gry_img = rgb2gray(rgb_img)
      game2 = re.findall("\/\D+\/", game)[-1][1:]
      epoch = re.findall("\d+", file_)[-1]
      np.save('grayscale/'+game2+"/game_{}/state_".format(i+1)+epoch+".npy", gry_img)
