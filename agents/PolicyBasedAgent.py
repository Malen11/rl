# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.policy_based import PolicyBased

class PolicyBasedAgent(object):

    def __init__(self,
                 action_num=2,
                 state_shape=None,
                 mlp_layers=None,
                 discount_factor=0.99,
                 learning_rate=0.00005,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 output_activation_func='tanh', 
                 output_kernel_initializer='RandomNormal',):
        self.use_raw = False
        
        # Create estimators
        self.algoritm = PolicyBased(
            num_state_params=state_shape[0],
            num_actions=action_num,
            hidden_units=np.full((mlp_layers[0]), mlp_layers[1]), 
            gamma=discount_factor, 
            learning_rate=learning_rate,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer
            )
           
        # Total timesteps
        self.total_t = 0
        
        # Total training step
        self.train_t = 0

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.algoritm.feed(state['obs'], action, reward)
        
        if done is True:
            self.train()
        
        self.total_t += 1
        
    def train(self):
        self.algoritm.train()
        self.train_t += 1
        
        self.algoritm.clear_memory()

    def step(self, state):
        return self.algoritm.get_action(state['obs'], state['legal_actions'])
    
    def eval_step(self, state):
        logits = self.algoritm.predict(state['obs']) 
        probs = self.algoritm.softmax(logits, state['legal_actions'])[0]
        best_action = np.argmax(probs)
        return best_action, probs
    
    def save_model(self, path):
        self.algoritm.save_model(path)
        
    def load_model(self, path):
        self.algoritm.load_model(path)