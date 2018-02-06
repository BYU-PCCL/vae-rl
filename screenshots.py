import gym
import numpy as np
import sys

game = sys.argv[1]
states_folder = sys.argv[2]
if states_folder[-1] != '/':
	states_folder += '/'

rewards_filepath, actions_filepath = None, None
if len(sys.argv) > 3:
	rewards_filepath = sys.argv[3]
	actions_filepath = sys.argv[4]

env = gym.make(game)
env.reset()
rewards, actions = [], []
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    actions.append(action)
    res = env.step(action) # take a random action
    rewards.append(res)
    if _ % 10 == 0:
        np.save(states_folder + "state_{}".format(_), res[0])

if rewards_filepath:
	np.save(rewards_filepath, rewards)
if actions_filepath:
	np.save(actions_filepath, actions)

