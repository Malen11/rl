# -*- coding: utf-8 -*-
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import device_lib

import os
import math

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.DDQNAgent import DDQNAgent
from agents.A2CAgent import A2CAgent

def main():
    tf.debugging.set_log_device_placement(True)
    print(device_lib.list_local_devices())
    print(tf.config.list_physical_devices('GPU'))
    print(tf.executing_eagerly())
    
    #names
    env_name = 'no-limit-holdem'
    dir_name = 'no_limit_holdem_a2c'
    test_name = 'test0'
    
    # Make environment
    env = rlcard.make(env_name, config={'seed': 0})
    eval_env = rlcard.make(env_name, config={'seed': 0})
    
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 10000
    evaluate_num = 1000
    episode_num = 100000
    
    # The intial memory size
    replay_memory_init_size = 100
    replay_memory_size = 10000
    
    # Train the agent every X steps
    train_every = 1
    
    # The paths for saving the logs and learning curves
    log_dir = './experiments/rl/'+dir_name+'_result'
    
    # Save model
    save_dir = 'models/rl/'+dir_name+'_result'
    
    # Set a global seed
    set_global_seed(0)
    
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent_test = A2CAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     mlp_layers=[8, 512],
                     discount_factor=0.9,
                     activation_func='relu', 
                     kernel_initializer='RandomNormal',
                     learning_rate=0.0001,
                     value_coef=0.9,)
    
    #agent_test.load_model(save_dir+'/'+test_name)
    agent_rand = RandomAgent(action_num=env.action_num)
    
    env.set_agents([agent_test, agent_rand])
    eval_env.set_agents([agent_test, agent_rand])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir+'/'+test_name)
    
    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent_test.feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            print('episode: ', episode)
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('PolicyBasedAgent')
    
    # Save model
    if not os.path.exists(save_dir+'/'+test_name):
        os.makedirs(save_dir+'/'+test_name)
        
    agent_test.save_model(save_dir+'/'+test_name)
    #saver = tf.train.Saver()
    #saver.save(sess, os.path.join(save_dir, 'model'))
        
if __name__ == '__main__':
    main()
