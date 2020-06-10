# -*- coding: utf-8 -*-
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import device_lib

import os
import math

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.DDQNAgent import DDQNAgent
from agents.A2CAgent import A2CAgent
from agents.testAgents import FoldAgent

def main():
    #tf.debugging.set_log_device_placement(True)
    print(device_lib.list_local_devices())
    print(tf.config.list_physical_devices('GPU'))
    #print(tf.executing_eagerly())
    
    #names
    env_name = 'no-limit-holdem'
    dir_name = 'no_limit_holdem_a2c_v2_est'
    test_name = 'test1'
    load_name = 'test1'
    
    #env nums
    env_num = 4
    
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 25000 // env_num
    evaluate_num = 10000
    episode_num = 1000000 // env_num
    
    # Train the agent every X steps
    train_every = 2048
    
    
    # Make environment
    env = rlcard.make(env_name, config={'seed': 0, 'env_num': 4})
    eval_env = rlcard.make(env_name, config={'seed': 0})    
    
    # The intial memory size
    #replay_memory_init_size = 100
    #replay_memory_size = 10000
    
    
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
                     
                     discount_factor=0.9,
                     lam=0.5,
                     
                     critic_mlp_layers=[4,256],
                     critic_activation_func='tanh', 
                     critic_kernel_initializer='glorot_uniform',
                     critic_learning_rate=0.0001,
                     critic_bacth_size=128,
                     
                     actor_mlp_layers=[4,256],
                     actor_activation_func='tanh', 
                     actor_kernel_initializer='glorot_uniform', 
                     actor_learning_rate=0.001,
                     actor_bacth_size=1024,
                     
                     entropy_coef=0.1,
                     entropy_decoy=math.pow(0.05/0.1, 1.0/episode_num),
                     #max_entropy_part=1.2,
                     
                     max_grad_norm = 1,)
    
    #agent_test.load_model(save_dir+'/'+test_name)
    agent_rand = RandomAgent(action_num=env.action_num)
    agent_fold = FoldAgent(action_num=env.action_num)
    
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
            
        if episode % (train_every // env_num) == 0:
            agent_test.train()
        
        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            print('episode: ', episode)
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('A2C')
    
    # Save model
    if not os.path.exists(save_dir+'/'+test_name):
        os.makedirs(save_dir+'/'+test_name)
        #os.makedirs(save_dir+'/'+test_name+'/critic')
        #os.makedirs(save_dir+'/'+test_name+'/actor')
        
    agent_test.save_model(save_dir+'/'+test_name)
    #saver = tf.train.Saver()
    #saver.save(sess, os.path.join(save_dir, 'model'))
        
if __name__ == '__main__':
    main()