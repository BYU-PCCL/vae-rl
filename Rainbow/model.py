import math
import torch
from torch import nn
from torch.nn import functional as F

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    if args.use_encoder == 0 or args.use_encoder == 2:
      self.view = 3136
      self.stride1, self.stride2 = 4, 2
      self.out1, self.out2, self.out3 = 8, 4, 3
    elif args.use_encoder == 1:
      self.view = 5376
      self.stride1, self.stride2 = 1, 1
      self.out1, self.out2, self.out3 = 1, 1, 1
    self.use_convcoord = args.use_convcord
    if self.use_convcord:
      self.conv1 = nn.Conv2d(args.history_length + 2, 32, self.out1, stride=self.stride1)
    else:
      self.conv1 = nn.Conv2d(args.history_length, 32, self.out1, stride=self.stride1)
    self.conv2 = nn.Conv2d(32, 64, self.out2, stride=self.stride2)
    self.conv3 = nn.Conv2d(64, 64, self.out3)
    self.device = args.device
    self.fc_h_v = NoisyLinear(self.view, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.view, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def add_coordconv(self, x):
    batch, _, h, w = x.size()
    row = torch.arange(w)

    j = row.repeat(h, 1)
    i = torch.t(j)
    i, j = i.to(device=self.device).type(torch.FloatTensor), j.to(device=self.device).type(torch.FloatTensor)
    i = 2. * (torch.reshape(i, (1, 1, h, w)) / (h - 1)) - 1
    j = 2. * (torch.reshape(j, (1, 1, h, w)) / (w - 1)) - 1
    if batch > 1:
      i, j = i.repeat(batch, 1, 1, 1), j.repeat(batch, 1, 1, 1)
    return torch.cat((x, i.to(device=self.device), j.to(device=self.device)), 1)

  def forward(self, x, log=False):
    if self.use_convcord:
      x = self.add_coordconv(x)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, self.view)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)
    if log:
      q = F.log_softmax(q, dim=2)
    else:
      q = F.softmax(q, dim=2)
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
