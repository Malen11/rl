# -*- coding: utf-8 -*-

import tensorflow as tf

class SimpleNeuralNetworkModel(tf.keras.Model):
    
    def __init__(self, 
                 num_input, 
                 hidden_units, 
                 num_output, 
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 output_activation_func='tanh', 
                 output_kernel_initializer='RandomNormal',
                 **kwargs):
        '''
        
        Инициализация модели сети

        Parameters
        ----------
        num_input : TYPE
            Количество параметров входного слоя.
        hidden_units : TYPE
            Массив размерности скрытых слоёв.
        num_output: TYPE
            Количество параметров выходного слоя.
        activation_func : TYPE, optional
            DESCRIPTION. The default is 'tanh'.
        kernel_initializer : TYPE, optional
            DESCRIPTION. The default is 'RandomNormal'.
        output_activation_func : TYPE, optional
            DESCRIPTION. The default is 'tanh'.
        output_kernel_initializer : TYPE, optional
            DESCRIPTION. The default is 'RandomNormal'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super(SimpleNeuralNetworkModel, self).__init__(**kwargs)
        
        self.num_input = num_input
        self.hidden_units = hidden_units
        self.num_output = num_output
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        self.output_activation_func = output_activation_func
        self.output_kernel_initializer = output_kernel_initializer
        
        #создание входного слоя сети
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_input,))
        
        #создание скрытых слоёв сети
        self.hidden_layers = []
        '''
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation=activation_func, kernel_initializer=kernel_initializer))
        '''
    
        for i in hidden_units:
            self.hidden_layers.append(SimpleNeuralNetworkLayerBlock(
                i, 
                activation_func=activation_func, 
                kernel_initializer=kernel_initializer))
            
        #создание выходного слоя сети
        self.output_layer = tf.keras.layers.Dense(
            num_output, 
            activation=output_activation_func, 
            kernel_initializer=output_kernel_initializer)
        
        
    @tf.function
    def call(self, inputs, training=None):
        '''
        Расчёт значений модели

        Parameters
        ----------
        inputs : TYPE
            Входные данные сети(состояние).
        training : TYPE
            режим тренировки.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        '''
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def get_config(self):
        
        config = super(SimpleNeuralNetworkModel, self).get_config()
        config.update({'num_input': self.num_input,
                       'hidden_units': self.hidden_units,
                       'num_output': self.num_output,
                       'activation_func': self.activation_func,
                       'kernel_initializer': self.kernel_initializer,
                       'output_activation_func': self.output_activation_func,
                       'output_kernel_initializer': self.output_kernel_initializer})
        return config
    
class ActorCriticNeuralNetworkModel(tf.keras.Model):
    
    def __init__(self, 
                 num_input, 
                 hidden_units, 
                 num_output, 
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='RandomNormal', 
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='RandomNormal',
                 **kwargs):
        '''
        
        Инициализация модели сети

        Parameters
        ----------
        num_input : TYPE
            Количество параметров входного слоя.
        hidden_units : TYPE
            Массив размерности скрытых слоёв.
        num_output: TYPE
            Количество параметров выходного слоя.
        activation_func : TYPE, optional
            DESCRIPTION. The default is 'tanh'.
        kernel_initializer : TYPE, optional
            DESCRIPTION. The default is 'RandomNormal'.
        output_activation_func : TYPE, optional
            DESCRIPTION. The default is 'tanh'.
        output_kernel_initializer : TYPE, optional
            DESCRIPTION. The default is 'RandomNormal'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super(ActorCriticNeuralNetworkModel, self).__init__(**kwargs)
        
        self.num_input = num_input
        self.hidden_units = hidden_units
        self.num_output = num_output
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        self.actor_activation_func = actor_activation_func
        self.actor_kernel_initializer = actor_kernel_initializer
        self.critic_activation_func = critic_activation_func
        self.critic_kernel_initializer = critic_kernel_initializer
        
        #создание входного слоя сети
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_input,))
        
        #создание скрытых слоёв сети
        self.hidden_layers = []
        '''
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation=activation_func, kernel_initializer=kernel_initializer))
        '''
    
        for i in hidden_units:
            self.hidden_layers.append(SimpleNeuralNetworkLayerBlock(
                i, 
                activation_func=activation_func, 
                kernel_initializer=kernel_initializer))
            
        #создание выходного слоя сети
        self.actor_output_layer = tf.keras.layers.Dense(
            num_output, 
            activation=actor_activation_func, 
            kernel_initializer=actor_kernel_initializer)
        
        #создание выходного слоя сети
        self.critic_output_layer = tf.keras.layers.Dense(
            1, 
            activation=critic_activation_func, 
            kernel_initializer=critic_kernel_initializer)
        
        
    @tf.function
    def call(self, inputs, training=None):
        '''
        Расчёт значений модели

        Parameters
        ----------
        inputs : TYPE
            Входные данные сети(состояние).
        training : TYPE
            режим тренировки.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        '''
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        policy_logits = self.actor_output_layer(x)
        values = self.critic_output_layer(x)
        #values = tf.squeeze(values,axis=-1)
        return policy_logits, values
    
    def get_config(self):
        
        config = super(SimpleNeuralNetworkModel, self).get_config()
        config.update({'num_input': self.num_input,
                       'hidden_units': self.hidden_units,
                       'num_output': self.num_output,
                       'activation_func': self.activation_func,
                       'kernel_initializer': self.kernel_initializer,
                       'output_activation_func': self.output_activation_func,
                       'output_kernel_initializer': self.output_kernel_initializer})
        return config
    
class SimpleNeuralNetworkLayerBlock(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation_func='tanh',
                 kernel_initializer='RandomNormal',
                 **kwargs
                 ):
      
        super(SimpleNeuralNetworkLayerBlock, self).__init__(**kwargs)
        
        self.units = units
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        
        self.dense_layer = tf.keras.layers.Dense(
            units, 
            kernel_initializer=kernel_initializer)
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.activation_layer = tf.keras.activations.get(activation_func)

    @tf.function
    def call(self, inputs, training=None):
      
        x = self.dense_layer(inputs)
        x = self.norm_layer(x, training)
        x = self.activation_layer(x)
        return x
    
    def get_config(self):
        
        config = super(SimpleNeuralNetworkLayerBlock, self).get_config()
        config.update({ 'units': self.units,
                        'activation_func': self.activation_func,
                        'kernel_initializer': self.kernel_initializer})
        
        return config