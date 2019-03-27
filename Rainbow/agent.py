import os
import numpy as np
import torch
from torch import optim
from model import DQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms # size of value distribution.
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount

    self.online_net = DQN(args, self.action_space).to(device=args.device) # greedily selects the action.
    if args.model and os.path.isfile(args.model):
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu')) # state_dict: python dictionary that maps each layer to its parameters.
    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device) # use to compute target q-values.
    self.update_target_net() # sets it to the parameters of the online network.
    self.target_net.train()
    for param in self.target_net.parameters(): # not updated through backpropagation.
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

  def reset_noise(self):
    self.online_net.reset_noise()

  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  def act_e_greedy(self, state, epsilon=0.001):
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    log_ps = self.online_net(states, log=True) # state log probabilities
    log_ps_a = log_ps[range(self.batch_size), actions] # get it for the batch size & current actions

    with torch.no_grad():
      pns = self.online_net(next_states) # next state probabilities
      # gets distribution 
      dns = self.support.expand_as(pns) * pns # expand_as: expands tensor to size of tensor
      argmax_indices_ns = dns.sum(2).argmax(1) # argmax selection
      self.target_net.reset_noise() 
      pns = self.target_net(next_states)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
      b = (Tz - self.Vmin) / self.delta_z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

    loss = -torch.sum(m * log_ps_a, 1)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()
    self.optimiser.step()
    mem.update_priorities(idxs, loss.detach().cpu().numpy()) # update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def save(self, path):
    torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
