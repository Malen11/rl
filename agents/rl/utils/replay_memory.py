# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

Replay = namedtuple('Replay', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory(object):
    
    def __init__(self,
                 max_replay_num=10000, 
                 min_replay_num=100):
        '''
        Класс для реализации памяти хранения игр

        Parameters
        ----------
        max_replay_num : TYPE, optional
            DESCRIPTION. The default is 10000.
        min_replay_num : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        '''
        self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        self.max_replay_num = max_replay_num
        self.min_replay_num = min_replay_num
        self.size = 0
        self.total_replays = 0
        
    def add_replay(self, replay):
        '''
        Добавить новую запись. Если достигнут предел по количеству записей, то
        самая старая запись будет очищена.

        Parameters
        ----------
        replay : Replay
            Кортеж состояния игры и результатов.

        Returns
        -------
        None.

        '''
        #
        if self.size > self.max_replay_num:
            for key in self.memory.keys():
                self.memory[key].pop(0)
            self.size -= 1
        #       
        for key in self.memory.keys():
            self.memory[key].append(replay[key])
        self.size += 1
        #
        self.total_replays += 1
        
    def clear(self):
        '''
        Очистить память

        Returns
        -------
        None.

        '''
        #
        for key in self.memory.keys():
            self.memory[key].clear()
        self.size = 0
        
    def get_random_samples(self, batch_size):
        '''
        Получить набор набор n случайных записей из памяти

        Parameters
        ----------
        batch_size : int
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        '''
        #выбираем batch_size записей
        ids = np.random.randint(low=0, high=self.size, size=batch_size)
        
        states = np.asarray([self.memory['state'][i] for i in ids])
        actions = np.asarray([self.memory['action'][i] for i in ids])
        rewards = np.asarray([self.memory['reward'][i] for i in ids])
        states_next = np.asarray([self.memory['next_state'][i] for i in ids])
        dones = np.asarray([self.memory['done'][i] for i in ids])
        
        return {'state': states, 'action':actions, 'reward': rewards, 'next_state': states_next, 'done': dones}