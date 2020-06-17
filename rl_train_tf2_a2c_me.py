# -*- coding: utf-8 -*-
import tensorflow as tf
#для исправления бага с совместимостью
#tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import device_lib

import os
import math
from multiprocessing import Process, Pipe

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.DDQNAgent import DDQNAgent
from agents.A2CAgent import A2CAgent
from agents.A2CQPGAgent import A2CQPGAgent
from agents.A2CLSTMAgent import A2CLSTMAgent
from agents.A2CLSTMQPGAgent import A2CLSTMQPGAgent
from agents.testAgents import FoldAgent

def main():
    #tf.debugging.set_log_device_placement(True)
    #print(device_lib.list_local_devices())
    #print(tf.config.list_physical_devices('GPU'))
    #print(tf.executing_eagerly())
    
    #names
    env_name = 'no-limit-holdem'
    dir_name = 'no_limit_holdem_a2c_v2_qpg'
    test_name = 'test_ddqn1000'
    load_dir = 'models/rl/no_limit_holdem_ddqn_result/test0'
    #env nums
    env_num = 1
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 25000 // env_num
    evaluate_num = 10000
    episode_num = 500000 // env_num
    # Train the agent every X steps
    train_every = 2048
    #random seed
    random_seed = 0
    
    # Make environment
    eval_env=rlcard.make(env_name, config={'seed': random_seed, 'env_num': env_num})
    
    # The paths for saving the logs and learning curves
    log_dir = './experiments/rl/'+dir_name+'_result'
    
    # Save model
    save_dir = 'models/rl/'+dir_name+'_result'
    
    # Set a global seed
    set_global_seed(random_seed)
    
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent_test = A2CLSTMQPGAgent(
                         action_num=eval_env.action_num,
                         state_shape=eval_env.state_shape,
                         
                         discount_factor=0.95,
                    
                         critic_lstm_layers=[1,51],
                         critic_mlp_layers=[3,51],
                         critic_activation_func='tanh', 
                         critic_kernel_initializer='glorot_uniform',
                         critic_learning_rate=0.001,
                         critic_bacth_size=128,
                         
                         actor_lstm_layers=[1,51],
                         actor_mlp_layers=[3,51],
                         actor_activation_func='tanh', 
                         actor_kernel_initializer='glorot_uniform', 
                         actor_learning_rate=0.0001,
                         actor_bacth_size=512,
                         
                         entropy_coef=0.5,
                         entropy_decoy=math.pow(0.1/0.5, 1.0/(episode_num//train_every)),
                         
                         max_grad_norm = 1,)
    #agent_test.load_model('models/rl/no_limit_holdem_a2c_v2_qpg_result/test_ddqn500')
    
    agent_ddqn = DDQNAgent(
                     action_num=eval_env.action_num,
                     state_shape=eval_env.state_shape,
                     epsilon_decay_coef=math.pow(0.05/1, 1.0/(episode_num//train_every)),) 
    #agent_ddqn.load_model(load_dir)
    eval_env=rlcard.make(env_name, config={'seed': random_seed, 'env_num': env_num})
    eval_env.set_agents([agent_test, agent_ddqn])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir+'/'+test_name)
    
    procs = []
    cons = []
    for i in range(env_num):
        parent_conn, child_conn = Pipe()
        cons.append(parent_conn)
        p = Process(target=run_worker, args=(child_conn, env_name, random_seed, train_every))
        p.start()
        procs.append(p)
    
    for episode in range(episode_num):
        for i in range(env_num):
            cons[i].send(['update', agent_test.get_weights()])
            
        for i in range(env_num):
            cons[i].send(['run'])
        
        for i in range(env_num):
            message = cons[i].recv()
            if message[0] == 'feed_batch':
                agent_test.feed_batch(message[1]) 
        
        agent_test.train() 
            
        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            print('episode: ', episode)
            logger.log_performance(episode, tournament(eval_env, evaluate_num)[0])            
    
    for i in range(env_num):
        cons[i].send(['end'])
    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot(dir_name)
    
    # Save model
    if not os.path.exists(save_dir+'/'+test_name):
        os.makedirs(save_dir+'/'+test_name)
        
    agent_test.save_model(save_dir+'/'+test_name)
 
def run_worker(conn, env_name, random_seed, episode_num):
    
    env = rlcard.make(env_name, config={'seed': random_seed})
    
    agent_test = A2CLSTMQPGAgent(
                         action_num=env.action_num,
                         state_shape=env.state_shape,
                         
                         discount_factor=0.95,
                    
                         critic_lstm_layers=[1,51],
                         critic_mlp_layers=[3,51],
                         critic_activation_func='tanh', 
                         critic_kernel_initializer='glorot_uniform',
                         critic_learning_rate=0.001,
                         critic_bacth_size=128,
                         
                         actor_lstm_layers=[1,51],
                         actor_mlp_layers=[3,51],
                         actor_activation_func='tanh', 
                         actor_kernel_initializer='glorot_uniform', 
                         actor_learning_rate=0.0001,
                         actor_bacth_size=512,
                         
                         entropy_coef=0.5,
                         entropy_decoy=math.pow(0.1/0.5, 1.0/(episode_num)),
                         
                         max_grad_norm = 1,)
    
    agent_ddqn = DDQNAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     epsilon_decay_coef=math.pow(0.05/1, 1.0/(episode_num)),) 
    
    env.set_agents([agent_test, agent_ddqn])
    
    message = conn.recv()
    while message[0] is not 'end': 
        if message[0] == 'run':
            print ('run')
            for episode in range(episode_num):
                # Generate data from the environment
                trajectories, _ = env.run(is_training=True)
        
                # Feed transitions into agent memory, and train the agent
                for ts in trajectories[0]:
                    agent_test.feed(ts)
            print ('run')
            conn.send(['feed_batch', agent_test.get_memory()])
        elif message[0] == 'update':
            print ('update')
            agent_test.set_weights(message[1])
            print ('update')
    
if __name__ == '__main__':
    main()
