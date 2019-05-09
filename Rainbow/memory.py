import random
from collections import namedtuple
import torch
import numpy as np
import cv2

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))

class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False
    self.sum_tree = np.zeros((2*size-1, ), dtype=np.float32)
    self.data = np.array([None]*size)
    self.max = 1
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)
  def update(self, index, value):
    self.sum_tree[index] = value
    self._propagate(index, value)
    self.max = max(value, self.max)
  def append(self, data, value):
    self.data[self.index] = data
    self.update(self.index + self.size - 1, value)
    self.index = (self.index + 1) % self.size
    self.full = self.full or self.index == 0
    self.max = max(value, self.max)
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])
  def find(self, value):
    index = self._retrieve(0, value)
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)
  def get(self, data_index):
    return self.data[data_index % self.size]
  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight
    self.priority_exponent = args.priority_exponent
    self.t = 0
    self.transitions = SegmentTree(capacity)
    self.encode_trans = args.encode_transitions
    if args.use_encoder == 0 or args.use_encoder == 2:
       self.xdim = 84
       self.ydim = 84
       self.zdim = 4
    elif args.use_encoder == 1:
       self.xdim = 1
       self.ydim = 84
    self.blank_trans = Transition(0, torch.zeros(self.zdim, self.xdim, self.ydim, dtype=torch.float32, device=self.device), None, 0, False)

  def append(self, state, action, reward, terminal):
    self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)
    self.t = 0 if terminal else self.t + 1

  def _get_transition(self, idx):
    transition = np.array([None]*2)
    transition[0] = self.transitions.get(idx)
    if transition[0].nonterminal:
      transition[1] = self.transitions.get(idx+1)
    else:
      transition[1] = self.blank_trans
    return transition

  def _get_sample_from_segment(self, segment, i):
    valid = False
    while not valid:
      sample = random.uniform(i * segment, (i + 1) * segment)
      prob, idx, tree_idx = self.transitions.find(sample)
      if (self.transitions.index - idx) % self.capacity > 1 and (idx - self.transitions.index) % self.capacity >= 2 and prob != 0:
        valid = True
    transition = self._get_transition(idx)
    state = transition[0].state
    next_state = transition[1].state
    action = torch.tensor([transition[0].action], dtype=torch.int64, device=self.device)
    R = torch.tensor([sum(self.discount ** n * transition[1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
    nonterminal = torch.tensor([transition[1].nonterminal], dtype=torch.float32, device=self.device)
    return prob, idx, tree_idx, state, action, R, next_state, nonterminal

  def sample(self, batch_size):
    p_total = self.transitions.total()
    segment = p_total / batch_size
    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    states, next_states, = torch.stack(states), torch.stack(next_states)
    actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
    probs = np.array(probs, dtype = np.float32)/p_total
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  def __iter__(self):
    self.current_idx = 0
    return self

  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    state = self.transitions.data[self.current_idx].state
    self.current_idx += 1
    return state
