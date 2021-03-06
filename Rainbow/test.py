import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from env import Env

Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10


def test(args, T, dqn, val_mem, env, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward

  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False
      action = dqn.act_e_greedy(state)
      state, reward, done = env.step(action)
      reward_sum += reward
      if args.render:
        env.render()
      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  env.train()
  for state in val_mem:
    T_Qs.append(dqn.evaluate_q(state))
  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    rewards.append(T_rewards)
    Qs.append(T_Qs)
    _plot_line(Ts, rewards, "Reward_{}".format(args.output_name), path='results')
    _plot_line(Ts, Qs, "Q_{}".format(args.output_name), path='results')
    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      dqn.save('results')
  return avg_reward, avg_Q, env

def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
