# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.a2c_v2_lstm_qpg import A2CLSTM

class A2CLSTMQPGAgent(object):

    def __init__(self,
                 action_num=2,
                 state_shape=None,
                 timesteps = 5,
                 
                 critic_lstm_layers=[1,256],
                 critic_mlp_layers=[3,256],
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='glorot_uniform',
                 critic_learning_rate=0.0001,
                 critic_bacth_size=128,
                 
                 actor_lstm_layers=[1,256],
                 actor_mlp_layers=[3,256],
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='glorot_uniform',  
                 actor_learning_rate=0.0001,
                 actor_bacth_size=2048,
                 
                 discount_factor=0.99,
                 #lam=0.5,
                 
                 entropy_coef=0.9,
                 entropy_decoy=1,
                 max_entropy_part=0.9,
                 
                 max_grad_norm = 0,
                 
                 min_reward=0,
                 max_reward=100):
        self.use_raw = False
        
        # Create estimators
        self.algoritm = A2CLSTM(
            num_state_params=state_shape[0],
            num_actions=action_num,
            
            critic_lstm_units=np.full((critic_lstm_layers[0]), critic_lstm_layers[1]), 
            critic_hidden_units=np.full((critic_mlp_layers[0]), critic_mlp_layers[1]), 
            critic_learning_rate=critic_learning_rate,
            critic_activation_func=critic_activation_func, 
            critic_kernel_initializer=critic_kernel_initializer,
            critic_bacth_size=critic_bacth_size,
            
            actor_lstm_units=np.full((actor_lstm_layers[0]), actor_lstm_layers[1]),
            actor_hidden_units=np.full((actor_mlp_layers[0]), actor_mlp_layers[1]),
            actor_learning_rate=actor_learning_rate, 
            actor_activation_func=actor_activation_func, 
            actor_kernel_initializer=actor_kernel_initializer,  
            actor_bacth_size=actor_bacth_size,
            
            gamma=discount_factor, 
            #lam = lam,
            
            entropy_coef=entropy_coef,
            entropy_decoy=entropy_decoy,
            max_entropy_part=max_entropy_part,
            
            max_grad_norm = max_grad_norm,
            )
           
        # Total timesteps
        self.total_t = 0
        
        # Total training step
        self.train_t = 0
        
        #normalization
        self.min_reward = 0
        self.max_reward = 100
        

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.algoritm.feed(
            state['obs'], 
            action,
            (reward-self.min_reward) / (self.max_reward-self.min_reward), 
            next_state['obs'], 
            done)
        
        self.total_t += 1
        
    def train(self):
        loss = self.algoritm.train()
        
        self.train_t += 1
        
        return loss

    def step(self, state):
        return self.algoritm.get_action(state['obs'], state['legal_actions'])
    
    def eval_step(self, state):
        
        self.algoritm.lstm.add_data(state['obs'])
        batch = [self.algoritm.lstm.get_data()]
        ts = tf.convert_to_tensor(batch)
        
        logits,_ = self.algoritm.predict(ts) 
        probs = self.algoritm.softmax(logits, state['legal_actions'])[0]
        best_action = np.argmax(probs)
        return best_action, probs
    
    def reset_lstm_memory(self):
        self.algoritm.reset_lstm_memory()
        
    def save_model(self, path):
        self.algoritm.save_model(path)
        
    def load_model(self, path):
        self.algoritm.load_model(path)