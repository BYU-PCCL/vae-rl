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
rewards, actions = [], []
for run in range(10):
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        actions.append(action)
        obs, rew, done, info = env.step(action) # Take a random action.
        rewards.append(rew)
        if done:
            break

        # Create a folder for the run if it doesn't already exist.
        states_folder_extended = states_folder + "game_{}/".format(run+1)
        os.makedirs(states_folder_extended, exist_ok=True)

        state_num = i/5
        if i % 5 == 0:
            np.save(states_folder_extended + "state_{}".format(state_num), obs)

    if rewards_filepath:
        np.save(rewards_filepath, rewards)
    if actions_filepath:
        np.save(actions_filepath, actions)
