# -*- coding: utf-8 -*-`
import random
import numpy as np
import tensorflow as tf
import os

from agents.rl.utils import ReplayMemory
from agents.rl.models.neural_network_models import SimpleNeuralNetworkModel
from pprint import pprint
class A2C(object):
    def __init__(self,
                 num_state_params, 
                 num_actions, 
                 
                 critic_hidden_units=[128],
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='glorot_uniform',
                 critic_learning_rate=0.0001,
                 critic_bacth_size=128,
                 
                 actor_hidden_units=[256],
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='glorot_uniform',  
                 actor_learning_rate=0.0001,
                 actor_bacth_size=2048,
                 
                 gamma=0.99, 
                 lam = 0.5,
                 
                 entropy_coef=0.9,
                 entropy_decoy=1,
                 max_entropy_part=0.9,
                 
                 max_grad_norm = 0,
                 ):
        
        #Параметры сети
        self.num_actions = num_actions
        self.num_state_params = num_state_params
        self.critic_hidden_units = critic_hidden_units
        self.actor_hidden_units = actor_hidden_units
        
        #параметры обучения
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate, decay=0.99, epsilon=1e-5)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate, decay=0.99, epsilon=1e-5)      
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.entropy_decoy = entropy_decoy
        self.max_entropy_part = max_entropy_part
        self.gamma = gamma
        self.lam = lam
        self.train_step = 0
        self.critic_bacth_size = critic_bacth_size
        self.actor_bacth_size = actor_bacth_size
        
        # Step_model that is used for sampling
        self.step_model_critic = SimpleNeuralNetworkModel(
            num_state_params,
            critic_hidden_units, 
            1,
            activation_func=critic_activation_func, 
            kernel_initializer=critic_kernel_initializer,
            output_activation_func=critic_activation_func, 
            output_kernel_initializer=critic_kernel_initializer, )
        
        self.step_model_actor = SimpleNeuralNetworkModel(
            num_state_params, 
            actor_hidden_units, 
            num_actions,
            activation_func=actor_activation_func, 
            kernel_initializer=actor_kernel_initializer,
            output_activation_func=actor_activation_func, 
            output_kernel_initializer=actor_kernel_initializer,)
        
        # Train model for training
        self.train_model_critic = SimpleNeuralNetworkModel(
            num_state_params,
            critic_hidden_units, 
            1,
            activation_func=critic_activation_func, 
            kernel_initializer=critic_kernel_initializer,
            output_activation_func=critic_activation_func, 
            output_kernel_initializer=critic_kernel_initializer, )
        
        self.train_model_actor = SimpleNeuralNetworkModel(
            num_state_params, 
            actor_hidden_units, 
            num_actions,
            activation_func=actor_activation_func, 
            kernel_initializer=actor_kernel_initializer,
            output_activation_func=actor_activation_func, 
            output_kernel_initializer=actor_kernel_initializer,)
        
        bag_shape = (1, self.num_state_params)
        bag_fix = tf.random.normal(bag_shape)
        _, _ = self._predict_train(bag_fix)
        _, _ = self.predict(bag_fix)
        
        self._update_step_model()
        
        #memory
        self.memory = ReplayMemory()
        
        #train memory
        self._mb_states = []
        self._mb_actions = []
        self._mb_returns = []
        self._mb_values = []
    
    def predict(self, inputs, training=False):
        policy_logits = self.predict_policy(inputs, training)
        values = self.predict_values(inputs, training)
        return policy_logits, values
    
    def predict_policy(self, inputs, training=False):
        policy_logits = self.step_model_actor(
            tf.convert_to_tensor(np.atleast_2d(inputs)))
        return policy_logits
    
    def predict_values(self, inputs, training=False):
        values = self.step_model_critic(
            tf.convert_to_tensor(np.atleast_2d(inputs)))
        values = tf.reshape(values, [-1])
        return values
    
    def _predict_train(self, inputs, training=False):
        policy_logits = self._predict_train_policy(inputs, training)
        values = self._predict_train_values(inputs, training)
        return policy_logits, values
    
    def _predict_train_policy(self, inputs, training=False):
        policy_logits = self.train_model_actor(
            tf.convert_to_tensor(np.atleast_2d(inputs)))
        return policy_logits
    
    def _predict_train_values(self, inputs, training=False):
        values = self.train_model_critic(
            tf.convert_to_tensor(np.atleast_2d(inputs)))
        values = tf.reshape(values, [-1])
        return values
    
    def feed(self, state, action, reward, next_state, done):
        replay = {
            'state': state,
            'action': action, 
            'reward': reward, 
            'next_state': next_state, 
            'done': done
            }
        self.memory.add_replay(replay)
        
    def clear_memory(self):
        self .memory.clear()
        
    def train(self):
        
        samples = self.memory.get_samples()
        
        est_values = self._predict_train_values(samples['state'])
        est_next_values = self._predict_train_values(samples['next_state'])
        
        #returns = self._returns_est(samples['reward'], samples['done'], est_values)
        #returns = self._returns(samples['reward'], samples['done'], est_values[-1])
        returns = self._general_advantage_estimates(
            samples['reward'],
            samples['done'], 
            est_values, 
            est_next_values, 
            self.lam)
        
        indices = [i for i in range(0, len(samples['state']))]
        random.shuffle(indices)
        
        states = np.asarray([samples['state'][i] for i in indices])
        actions = np.asarray([samples['action'][i] for i in indices])
        returns = np.asarray([returns[i] for i in indices])
        
        critic_loss = self._critic_train(states, returns)
        policy_loss, entropy_loss, policy_entropy_loss = self._actor_train(states, actions, returns)

        self._update_step_model()
        self.entropy_coef *= self.entropy_decoy
        self.train_step += 1

        test_logit, test_value = self.predict(samples['state'][0])
        print(test_logit)
        print("------------------------")
        print(test_value)
        print("------------------------")
        loss = [critic_loss, policy_loss, entropy_loss, policy_entropy_loss]
        print(loss)
        print("========================")
        
        self.clear_memory()
        
        return loss
    
    def _critic_train(self, states, returns):

        critic_loss_list = []

        for start in range(0, self.memory.size, self.critic_bacth_size):

            if self.memory.size > start+self.critic_bacth_size:

                indices = range(start, start+self.critic_bacth_size)
                self._mb_states=states[indices]
                self._mb_returns=returns[indices]
                
                critic_loss = self._critic_train_step()

                critic_loss_list.append(critic_loss.numpy())

        return critic_loss_list
    
    @tf.function
    def _critic_train_step(self):
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
                        
            #logits - вектор необработанных (ненормализованных) предсказаний, 
            #которые генерирует модель классификации
            values = self._predict_train_values(self._mb_states)
            value_loss = self._value_loss(self._mb_returns, values)
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        value_weights = self.train_model_critic.trainable_weights
        value_gradients = tape.gradient(value_loss, value_weights)
        
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            value_gradients, _ = tf.clip_by_global_norm(value_gradients, self.max_grad_norm)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.critic_optimizer.apply_gradients(zip(value_gradients, value_weights))
        
        return value_loss
    
    def _actor_train(self, states, actions, returns):

        entropy_loss_list = []
        policy_loss_list = []
        policy_entropy_loss_list = []
            
        values = self._predict_train_values(states).numpy()

        for start in range(0, self.memory.size, self.actor_bacth_size):

            if self.memory.size > start+self.actor_bacth_size:

                indices = range(start, start+self.actor_bacth_size)
                self._mb_states=states[indices]
                self._mb_actions=actions[indices]
                self._mb_returns=returns[indices]
                self._mb_values=values[indices]

                policy_loss, entropy_loss, policy_entropy_loss = self._actor_train_step()

                entropy_loss_list.append(entropy_loss.numpy())
                policy_loss_list.append(policy_loss.numpy())
                policy_entropy_loss_list.append(policy_entropy_loss.numpy())

        return policy_loss_list, entropy_loss_list, policy_entropy_loss_list 
    
    @tf.function
    def _actor_train_step(self):
        
        with tf.GradientTape() as tape:
                        
            policy_logits = self.train_model_actor(self._mb_states)
            advantages = self._advantages(self._mb_returns, self._mb_values)
            policy_loss = self._policy_loss(self._mb_actions, advantages, policy_logits)
            entropy_loss = self._entropy_loss(policy_logits)
            #clip_entropy_loss = tf.minimum(entropy_loss*self.max_entropy_part, entropy_loss)
            policy_entropy_loss = policy_loss - self.entropy_coef * entropy_loss
            
        policy_weights = self.train_model_actor.trainable_weights
        policy_gradients = tape.gradient(policy_entropy_loss, policy_weights)
        
        if self.max_grad_norm is not None:
            policy_gradients, _ = tf.clip_by_global_norm(policy_gradients, self.max_grad_norm)
            
        self.actor_optimizer.apply_gradients(zip(policy_gradients, policy_weights))
        return policy_loss, entropy_loss, policy_entropy_loss
    
    def _update_step_model(self):
        self.step_model_critic.set_weights(self.train_model_critic.get_weights()) 
        self.step_model_actor.set_weights(self.train_model_actor.get_weights()) 
        
    def get_action(self, state, legal_actions):
        
        logits = self.predict_policy(np.atleast_2d(state))
        probs = self.softmax(logits, legal_actions)[0]
        selected_action = np.random.choice(self.num_actions, p=probs)
        
        return selected_action
    
    def softmax(self, logits, legal_actions=None):
        
        probs = tf.keras.activations.softmax(logits)
        probs = probs * tf.reduce_sum(tf.one_hot(legal_actions, self.num_actions), axis=0)
        probs = probs / tf.reduce_sum(probs, axis=1)
            
        return probs.numpy()
    
    def argmax(self, logits, legal_actions=None):
        
        probs = self.softmax(logits, legal_actions)
        
        ts = tf.Variable(probs)
        return tf.math.argmax(ts, axis=1).numpy()
    
    def _value_loss(self, returns, values):
        # Value loss is typically MSE between value estimates and rewards.
        return tf.math.reduce_mean(tf.keras.losses.MSE(returns, values))
      
    def _policy_loss(self, actions, advantages, logits):
        
        actions = tf.cast(actions, dtype='int32')
        
        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        #weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
             labels=actions, 
             logits=logits)
        
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        #policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # We want to minimize policy 
        policy_loss = tf.math.reduce_mean(cross_entropy * advantages)
        return policy_loss
        #return policy_loss
      
    def _entropy_loss(self, logits):
        
        # Entropy loss can be calculated as cross-entropy over itself.
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        
        return tf.math.reduce_mean(entropy_loss)
    
    def _advantages(self, returns, values):
        advantages = returns-values
        return advantages
    
    def _returns(self, rewards, dones, last_value):
        returns = np.zeros_like(rewards)
        next_value = last_value
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * next_value
            next_value = returns[t]
        return returns
    
    def _returns_est(self, rewards, dones, next_values):
        returns = rewards + (1 - dones) * self.gamma * next_values
        return returns
      
    def _general_advantage_estimates(self, rewards, dones, values, next_values, lam):
        ### GENERALIZED ADVANTAGE ESTIMATION
        # discount/bootstrap off value fn
        # We create mb_returns and mb_advantages
        # mb_returns will contain Advantage + value
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        T = len(rewards)
        
        # From last step to first step
        for t in reversed(range(T)):
            # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
            delta = rewards[t] + (1- dones[t]) * self.gamma * next_values[t] - values[t]
            # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
            advantages[t] = lastgaelam = delta + self.gamma * lam * (1- dones[t]) * lastgaelam
        # Returns
        returns = advantages + values
        return returns
  
    def _normalize(self, data):
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0: std = 1
            
        norm = (data - mean) / (std)
        
        return norm

    def save_model(self, path):

        if not os.path.exists(path+'/critic'):
            os.makedirs(path+'/critic')
            os.makedirs(path+'/actor')

        #bag_shape = (1, self.num_state_params)
        #bag_fix = tf.random.normal(bag_shape)
        #self.train_model_critic.predict(bag_fix)
        #self.train_model_actor.predict(bag_fix)
        self.train_model_critic.save(path+'/critic', save_format="tf")
        self.train_model_actor.save(path+'/actor', save_format="tf")
        
    def load_model(self, path):
        self.train_model_critic = tf.keras.models.load_model(path+'/critic')
        self.train_model_actor = tf.keras.models.load_model(path+'/actor')
        self.step_model_critic = tf.keras.models.load_model(path+'/critic')
        self.step_model_actor = tf.keras.models.load_model(path+'/actor')
        self._update_step_model()
