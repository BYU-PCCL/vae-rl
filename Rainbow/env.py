from collections import deque
import random
import atari_py
import torch
import cv2
import numpy as np
import os

from vlae.vladder import VLadder
from vlae.dataset import AtariDataset

class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0
    self.life_termination = False
    self.window = args.history_length
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True

    self.useVLAE = args.use_encoder
    if self.useVLAE:
      self.dataset = AtariDataset(db_path='')
      self.network = VLadder(self.dataset, file_path='vlae/models/', name='', reg='kl', batch_size=100, restart=False)

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _get_stateVLAE(self):
    state = self.ale.getScreenRGB()
    state = cv2.resize(state, (96, 96), interpolation=cv2.INTER_LINEAR)
    state = self.network.get_latent_codes(np.reshape(state,(1,96,96,3)))
    state = torch.tensor(state, dtype=torch.float32, device=self.device)
    return state

  def _reset_buffer(self):
    for _ in range(self.window):
      if self.useVLAE:
        self.state_buffer.append(torch.zeros(1, 40, device=self.device))
      else:
        self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False
      self.ale.act(0)
    else:
      self._reset_buffer()
      self.ale.reset_game()
      for _ in range(random.randrange(30)):
        self.ale.act(0)
        if self.ale.game_over():
          self.ale.reset_game()
    if self.useVLAE:
      observation = self._get_stateVLAE()
    else:
      observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    if self.useVLAE:
      frame_buffer = torch.zeros(2, 1, 40, device=self.device)
    else:
      frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        if self.useVLAE:
          frame_buffer[0] = self._get_stateVLAE()
        else:
          frame_buffer[0] = self._get_state()
      elif t == 3:
        if self.useVLAE:
          frame_buffer[1] = self._get_stateVLAE()
        else:
          frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:
        self.life_termination = not done
        done = True
      self.lives = lives
    return torch.stack(list(self.state_buffer), 0), reward, done

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
