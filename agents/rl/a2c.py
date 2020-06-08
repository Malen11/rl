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
                 max_grad_norm = 0
                 ):
        
        #Параметры сети
        self.num_actions = num_actions
        self.num_state_params = num_state_params
        self.hidden_units = hidden_units
        
        #параметры обучения
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, decay=0.99, epsilon=1e-5)   
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.train_step = 0
        
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
        self.last_value = 0
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
        
        _,est_value = self.predict(self.next_states[-1])
        last_value = est_value.numpy()[0][0]
        
        self.size += 1
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.size = 0
     
    @tf.function   
    def train(self):

        states = self.states
        actions = self.actions
        rewards = self.rewards
        #rewards = self._normalize(self.rewards)
        dones = self.dones
        next_states = self.next_states
        
        returns = self._returns(rewards, dones, self.last_value)
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
                        
            #logits - вектор необработанных (ненормализованных) предсказаний, 
            #которые генерирует модель классификации
            policy_logits, values = self.predict(self.states)
            advantages = self._advantages(returns, values)
            
            policy_loss = self._policy_loss(actions,advantages, policy_logits)
            value_loss = self._value_loss(returns, values)
            
        '''
        policy_logits0 = policy_logits.numpy()
        values0 = values.numpy()
        '''
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        variables = self.step_model.trainable_weights
        gradients = tape.gradient([policy_loss, value_loss], variables)
        
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.train_step += 1

        return [policy_loss,value_loss]
    
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
    
    '''
    def _loss(self, values, estimated_values):
        estimated_values = tf.reshape(estimated_values, [-1])
        policy_logits1 = policy_logits.numpy()
        est_values2 =est_values.numpy()
        
        # Compute the loss value for this minibatch.     
        advantages = self._advantages(
            self.rewards, est_values, self.dones, last_state_value)
        value_loss = self._value_loss(values, est_values)
        policy_loss  = self._policy_loss(policy_logits, self.actions, advantages)
        entropy_loss = self._entropy_loss(policy_logits)
        
        loss_values = tf.math.reduce_mean(
            policy_loss - entropy_loss * self.entropy_coef + value_loss * self.value_coef)
        
        t1 = policy_loss.numpy()
        t2 =entropy_loss.numpy()
        t3=self.entropy_coef
        t4=value_loss.numpy()
        t5=self.value_coef
        t0=loss_values.numpy()'''
    
    def _value_loss(self, returns, values):
        # Value loss is typically MSE between value estimates and rewards.
        return tf.math.reduce_mean(self.value_coef*tf.keras.losses.MSE(returns, values))
      
    def _policy_loss(self, actions,advantages, logits):
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
        return tf.math.reduce_mean(policy_loss - self.entropy_coef * entropy_loss)
        #return policy_loss - self.entropy_coef * entropy_loss
      
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
    
    def _normalize(self, data):
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            std = 1
            
        norm = (data - mean) / (std)
        
        return norm
    
    def save_model(self, path):
        #shape = (1, self.num_state_params)
        #bag_fix = tf.random.normal(shape)
        #self.p_net.predict(bag_fix)
        self.train_model.save(path, save_format="tf")
        
    def load_model(self, path):
       self.train_model = tf.keras.models.load_model(path)