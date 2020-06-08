# -*- coding: utf-8 -*-`

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.models.neural_network_models import ActorCriticNeuralNetworkModel
from pprint import pprint

class A2C(object):

    def __init__(self,
                 num_state_params, 
                 num_actions, 
                 hidden_units, 
                 gamma=0.99, 
                 learning_rate=0.00005,
                 value_coef=0.9,
                 entropy_coef=0.01,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='RandomNormal', 
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='RandomNormal',
                 ):
        
        #Параметры сети
        self.num_actions = num_actions
        self.num_state_params = num_state_params
        self.hidden_units = hidden_units
        
        #параметры обучения
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)   
        self.train_step = 0
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        #параметры предсказания
        self.gamma = gamma
        
        # Step_model that is used for sampling
        self.step_model = ActorCriticNeuralNetworkModel(
            num_state_params,
            hidden_units, 
            num_actions,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer,
            actor_activation_func=actor_activation_func, 
            actor_kernel_initializer=actor_kernel_initializer, 
            critic_activation_func=critic_activation_func, 
            critic_kernel_initializer=critic_kernel_initializer,)
        self.step_model.compile(
            optimizer=self.optimizer,
            # define separate losses for policy logits and value estimate
            loss=[self._policy_loss, self._value_loss]
        )
        # Train model for training
        self.train_model = ActorCriticNeuralNetworkModel(
            num_state_params,
            hidden_units, 
            num_actions,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer,
            actor_activation_func=actor_activation_func, 
            actor_kernel_initializer=actor_kernel_initializer, 
            critic_activation_func=critic_activation_func, 
            critic_kernel_initializer=critic_kernel_initializer,)
        
        #memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.size = 0
    
    def predict(self, inputs, training=False):
        ts_inputs = tf.convert_to_tensor(np.atleast_2d(inputs))
        policy_logits, values = self.step_model(ts_inputs)
        return policy_logits, values
    
    def feed(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.size += 1
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.size = 0
        
    def train(self):

        states = self.states
        actions = np.array(self.actions)
        rewards = self._normalize(self.rewards)
        dones = self.dones
        next_states=self.next_states
        
        _,est_value = self.predict(self.next_states[-1])
        last_value = est_value.numpy()[0][0]
        returns = self._returns(rewards, dones, last_value)
        
        logits, values = self.predict(self.states)
        advantages = np.reshape(self._advantages(returns, values), np.reshape(-1,1))
        
        actions_advantages = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)
        
        # performs a full training step on the collected batch
        # note: no need to mess around with gradients, Keras API handles it
        losses = self.step_model.train_on_batch(np.reshape(self.states,(-1, 54)), [actions_advantages, returns])
        
        t1 = logits.numpy()
        t2 = values.numpy()
        
        self.train_step += 1

        return losses
    
    def get_action(self, state, legal_actions):
        
        logits,_ = self.predict(np.atleast_2d(state))
        probs = self.softmax(logits, legal_actions)[0]
        selected_action = np.random.choice(self.num_actions, p=probs)
        
        return selected_action
    
    def softmax(self, logits, legal_actions=None):
        
        probs = tf.keras.activations.softmax(logits)
        probs = probs * tf.reduce_sum(tf.one_hot(legal_actions, self.num_actions,dtype='double'), axis=0)
        probs = probs / tf.reduce_sum(probs, axis=1)
            
        return probs.numpy()
    
    def argmax(self, logits, legal_actions=None):
        
        probs = self.softmax(logits, legal_actions)
        
        ts = tf.Variable(probs)
        return tf.math.argmax(ts, axis=1).numpy()
    
    def _normalize(self, data):
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1
            
        norm = (data - mean) / (std)
        
        return norm
    
    def _value_loss(self, returns, values):
        # Value loss is typically MSE between value estimates and rewards.
        value_loss = self.value_coef*tf.keras.losses.MSE(returns, values)
        return value_loss
      
    def _policy_loss(self, actions_advantages, logits):
        
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(actions_advantages, 2, axis=-1)
        
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=actions, 
        #     logits=logits)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = self._entropy_loss(logits)
        # We want to minimize policy 
        policy_loss -= self.entropy_coef * entropy_loss
        
        #return policy_loss - self.entropy_coef * entropy_loss
        return policy_loss
      
    def _entropy_loss(self, logits):
        
        # Entropy loss can be calculated as cross-entropy over itself.
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return entropy_loss

    def _returns(self, rewards, dones, last_value):
        returns = np.zeros_like(rewards)
        
        next_value = last_value
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * next_value
            next_value = returns[t]
            
        returns = np.reshape(returns, (-1, 1))
        return returns
    
    def _advantages(self, returns, values):
        
        advantages = returns-values
            
        return advantages
    
    def save_model(self, path):
        #shape = (1, self.num_state_params)
        #bag_fix = tf.random.normal(shape)
        #self.p_net.predict(bag_fix)
        self.p_net.save(path, save_format="tf")
        
    def load_model(self, path):
       self.p_net = tf.keras.models.load_model(path)