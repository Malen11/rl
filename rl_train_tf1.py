# -*- coding: utf-8 -*-
''' Q-Table learning
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os

import rlcard
from rlcard.agents import DQNAgent, RandomAgent,NFSPAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.DDQNAgent import DDQNAgent

def main():
    # Make environment
    env = rlcard.make('no-limit-holdem', config={'seed': 0})
    eval_env = rlcard.make('no-limit-holdem', config={'seed': 0})
    
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 5000
    evaluate_num = 10000
    episode_num = 100000
    
    # The intial memory size
    memory_init_size = 256
    
    # Train the agent every X steps
    train_every = 1
    
    # The paths for saving the logs and learning curves
    log_dir = './experiments/no_limit_holdem_nfsp_result/'
    save_dir = 'models/no_limit_holdem_nfsp_result'
    
    # Set a global seed
    set_global_seed(0)
    
    with tf.Session() as sess:
    
        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
    
        # Set up the agents
        '''
        agent_test = DQNAgent(sess,
                         scope='dqn',
                         action_num=env.action_num,
                         replay_memory_init_size=memory_init_size,
                         train_every=train_every,
                         state_shape=env.state_shape,
                         mlp_layers=[10,100])
        '''
        agent_test = NFSPAgent(sess,
                          scope='nfsp',
                          action_num=env.action_num,
                          state_shape=env.state_shape,
                          hidden_layers_sizes=[512,512],
                          anticipatory_param=0.1,
                          min_buffer_size_to_learn=memory_init_size,
                          q_replay_memory_init_size=memory_init_size,
                          train_every = train_every,
                          q_train_every=train_every,
                          q_mlp_layers=[512,512])
        
        agent_rand = RandomAgent(action_num=env.action_num)
        env.set_agents([agent_test, agent_rand])
        eval_env.set_agents([agent_test, agent_rand])
    
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
    
        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)
        
        for episode in range(episode_num):
    
            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)
    
            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent_test.feed(ts)
    
            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
    
        # Close files in the logger
        logger.close_files()
    
        # Plot the learning curve
        logger.plot('q_table')
        
        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, 'model'))
        
if __name__ == '__main__':
    main()