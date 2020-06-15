# -*- coding: utf-8 -*-
"""
An example of loading a pre-trained NFSP model on Leduc Hold'em
"""

import tensorflow
import os
import rlcard

from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import set_global_seed, tournament
from pprint import pprint

from agents.testAgents import FoldAgent, CallAgent, RiseAgent

# Make environment
env = rlcard.make('no-limit-holdem', config={'seed': 1, 'allow_raw_data':True})
episode_num = 3

# Set a global seed
set_global_seed(1)
    
# Set up agents
agent_rand = RandomAgent(action_num=env.action_num)
agent_fold = FoldAgent(action_num=env.action_num)
agent_call = CallAgent(action_num=env.action_num)
agent_rise = RiseAgent(action_num=env.action_num)
agent_DQN = None

# Load pretrained model
graph = tensorflow.Graph()
sess = tensorflow.Session(graph=graph)
with graph.as_default():
    agent_DQN = DQNAgent(sess,
                    scope='dqn',
                    action_num=env.action_num,
                    state_shape=env.state_shape,
                    mlp_layers=[10,10])
    
# We have a pretrained model here. Change the path for your model.
check_point_path = 'models/no_limit_holdem_dqn_result'

with sess.as_default():
    with graph.as_default():
        saver = tensorflow.train.Saver()
        saver.restore(sess, tensorflow.train.latest_checkpoint(check_point_path))

env.set_agents([agent_DQN, agent_rand])

# Evaluate the performance. Play with random agents.
evaluate_num = 1000
rewards = tournament(env, evaluate_num)
print('Average reward against random agent: ', rewards[0])

'''
for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)
    
    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    for ts in trajectories[1]:
        print('\n\nState: {}, \n\nAction: {}, \n\nReward: {}, \n\nNext State: {}, \n\nDone: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))
'''