import gym
import numpy as np
import sys
import os

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
for run in range(10):
    for i in range(1000):
        # env.render()
        action = env.action_space.sample()
        actions.append(action)
        res = env.step(action) # take a random action
        rewards.append(res)
        
        # create a folder for the run if it doesn't already exist
        states_folder_extended = states_folder + "run_{}/".format(run+1)
        os.makedirs(states_folder_extended, exist_ok=True)

        state_num = i/10
        if i % 10 == 0:
            np.save(states_folder_extended + "state_{}".format(state_num), res[0])

    if rewards_filepath:
        np.save(rewards_filepath, rewards)
    if actions_filepath:
        np.save(actions_filepath, actions)

