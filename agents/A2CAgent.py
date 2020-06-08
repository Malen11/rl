# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.a2c import A2C

class A2CAgent(object):

    def __init__(self,
                 action_num=2,
                 state_shape=None,
                 mlp_layers=None,
                 discount_factor=0.99,
                 learning_rate=0.00005,
                 value_coef=0.9,
                 entropy_coef=0.9,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='RandomNormal', 
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='RandomNormal',
                 max_grad_norm = 0,
                 min_reward=0,
                 max_reward=100):
        self.use_raw = False
        
        # Create estimators
        self.algoritm = A2C(
            num_state_params=state_shape[0],
            num_actions=action_num,
            hidden_units=np.full((mlp_layers[0]), mlp_layers[1]), 
            gamma=discount_factor, 
            learning_rate=learning_rate,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer,
            actor_activation_func=actor_activation_func, 
            actor_kernel_initializer=actor_kernel_initializer, 
            critic_activation_func=critic_activation_func, 
            critic_kernel_initializer=critic_kernel_initializer,
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
        
        if done is True:
            self.train()
        
        self.total_t += 1
        
    def train(self):
        loss = self.algoritm.train()
        loss = [loss[0].numpy(), loss[1].numpy()]
        self.train_t += 1
        
        self.algoritm.clear_memory()
        return loss

    def step(self, state):
        return self.algoritm.get_action(state['obs'], state['legal_actions'])
    
    def eval_step(self, state):
        logits,_ = self.algoritm.predict(state['obs']) 
        probs = self.algoritm.softmax(logits, state['legal_actions'])[0]
        best_action = np.argmax(probs)
        return best_action, probs
    
    def save_model(self, path):
        self.algoritm.save_model(path)
        
    def load_model(self, path):
        self.algoritm.load_model(path)
