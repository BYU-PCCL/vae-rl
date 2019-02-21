import argparse
from datetime import datetime
import random
import torch
from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigma', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='omega', help='Prioritised experience replay exponent (originally denoted alpha)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='beta', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='tao', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='eta', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='eps', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--use-encoder', action='store_false', help='Use VLAE')

args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False
else:
  args.device = torch.device('cpu')

def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

env = Env(args)
env.train()
action_space = env.action_space()

dqn = Agent(args, env)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False
  next_state, _, done = env.step(random.randint(0, action_space - 1))
  val_mem.append(state, None, None, done)
  state = next_state
  T += 1
if args.evaluate:
  dqn.eval()
  # avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)
  avg_reward, avg_Q, env = test(args, 0, dqn, val_mem, env, evaluate=True)
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  dqn.train()
  T, done = 0, True
  while T < args.T_max:
    if done:
      state, done = env.reset(), False
    if T % args.replay_frequency == 0:
      dqn.reset_noise()
    action = dqn.act(state)
    next_state, reward, done = env.step(action)
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)
    mem.append(state, action, reward, done)
    T += 1
    if T % args.log_interval == 0:
      log('T = ' + str(T) + ' / ' + str(args.T_max))
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
      if T % args.replay_frequency == 0:
        dqn.learn(mem)
      if T % args.evaluation_interval == 0:
        dqn.eval()
        # avg_reward, av_Q = test(args, T, dqn, val_mem)
        avg_reward, avg_Q, env = test(args, T, dqn, val_mem, env)
        log('T = '+str(T)+' / '+str(args.T_max)+' | Avg. reward: '+str(avg_reward)+' | Avg. Q: '+str(avg_Q))
        dqn.train()
      if T % args.target_update == 0:
        dqn.update_target_net()
    state = next_state
env.close()
