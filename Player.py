# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:11:43 2016
Implemented in Python 3.5.2
Author: Yun-Jhong Wu
"""

import numpy as np
import os

from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, dot
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import RMSprop
from typing import Tuple


def getModel(input_shape):
    Qfunc = Sequential()
    action = Input(shape=(Player.NUM_ACTIONS,))
    screen = Input(shape=input_shape)
    Qfunc.add(Conv2D(32, (8, 8), strides=(4, 4), 
                     activation='relu', padding="same", 
                     input_shape=input_shape,
                     data_format="channels_first"))
    Qfunc.add(Conv2D(64, (4, 4), strides=(2, 2), 
                     activation='relu', padding="same", 
                     data_format="channels_first"))
    Qfunc.add(Conv2D(64, (4, 4), activation='relu', padding="same",
                     data_format="channels_first"))
    Qfunc.add(MaxPooling2D(pool_size=(2, 2)))
    Qfunc.add(Flatten())
    Qfunc.add(Dense(128, activation='relu'))
    Qfunc.add(Dense(128, activation='relu'))
    Qfunc.add(Dense(Player.NUM_ACTIONS))
    
    reward = Qfunc(screen)
    
    model = Model(inputs=[screen, action], 
                  outputs=dot([reward, action], axes=[1, 1]))
    return Qfunc, model


class Player:
    BANDWIDTH = 5
    INPUT_SHAPE = (80, 80)
    MEMORY_CAPACITY = 1000000

    BATCH_SIZE = 32    
    INIT_EPSILON = 1.0
    FINAL_EPSILON = 0.1  
    DECAY_EPSILON = (INIT_EPSILON - FINAL_EPSILON) / 500000
    GAMMA = 0.99
    Q_UPDATE_FREQUENCY = 5000
    NUM_ACTIONS = 3

    def __init__(self, learning: bool=True, 
                 load_model: bool=True, 
                 model_path: str="serialized/model.json",
                 weights_path: str="serialized/weights.h5"):
                         
        if not os.path.isdir("log"):
            os.mkdir("log")                

        if not os.path.isdir("serialized"):
            os.mkdir("serialized")        
            
        self.niters = 0
        self.avgQ = 0
        self.loss = 0
        self.memory_size = 0
        self.memory = []        
        self.cacheQ = {}
        
        self.model_path = model_path
        self.weights_path = weights_path
        self.learning = learning
        self.load_model = load_model
        self.epsilon = Player.INIT_EPSILON if self.learning else 0
        self.initModel()


    def remember(self, snapshot: Tuple): # (frame, action, reward, is_playing)
        self.memory.append(snapshot)
        
        if self.memory_size > Player.BANDWIDTH * 2:
            if self.learning:
                if self.memory_size > Player.MEMORY_CAPACITY * 1.05:
                    self.memory_size = Player.MEMORY_CAPACITY
                    self.memory = self.memory[-Player.MEMORY_CAPACITY:]
                    self.cacheQ = {}

                else:
                    self.memory_size += 1

            else:
                self.memory_size = Player.BANDWIDTH * 2
                self.memory = self.memory[-self.memory_size:]
        else:
            self.memory_size += 1
   
     
    def getAction(self, *args):
        
        if self.memory_size < Player.BANDWIDTH * 2 or (self.learning and np.random.binomial(1, self.epsilon)):
            return np.random.randint(0, Player.NUM_ACTIONS)

        else:
            qscore = self.Qfunc.predict(self.getCurrent())
            self.avgQ = np.mean(qscore)
 
            return np.argmax(qscore)
 

    def getCurrent(self):

        return np.stack([self.memory[i][0] for i in range(-Player.BANDWIDTH, 0)])[None, :]


    def getTransition(self, cache):
        is_playing = all(cache[i + Player.BANDWIDTH][3] for i in range(Player.BANDWIDTH))
        
        trans = (np.stack([cache[i][0] for i in range(Player.BANDWIDTH)]),
                 np.stack([cache[i + Player.BANDWIDTH][0] for i in range(Player.BANDWIDTH)]),
                 (cache[Player.BANDWIDTH][1] == np.arange(Player.NUM_ACTIONS)) * 1, 
                 (cache[-1][2] > cache[Player.BANDWIDTH][2]) - (not is_playing),
                 is_playing)
                 
        return trans


    def getBatch(self):
        sample = np.random.randint(Player.BANDWIDTH * 2, self.memory_size, Player.BATCH_SIZE)
        trans = [self.getTransition(self.memory[idx - Player.BANDWIDTH * 2:idx]) for idx in sample]
        screen = np.stack([t[0] for t in trans])
        action = np.stack([t[2] for t in trans])
        reward = np.stack([t[3] for t in trans])

        for i, idx in enumerate(sample):
            if trans[i][4]:
                if idx not in self.cacheQ:                    
                    self.cacheQ[idx] = Player.GAMMA * self.Qtarget.predict(trans[i][1][None, :]).ravel()
                    
                reward[i] += self.cacheQ[idx][np.argmax(self.Qfunc.predict(trans[i][1][None, :]).ravel())]

        return screen, action, reward
    
    
    def train(self, log: bool=True):
        if self.learning:
            if self.epsilon > Player.FINAL_EPSILON:
                self.epsilon -= Player.DECAY_EPSILON

            screen, action, reward = self.getBatch()
            loss = self.model.train_on_batch([screen, action], reward) 
            self.loss = loss.take(0) / Player.BATCH_SIZE
            self.niters += 1

            print("Training: Iteration {0}: loss = {1:.4f}, avg. Q = {2:.4f}, epsilon = {3:.4f}".format(self.niters, self.loss, self.avgQ, self.epsilon))

            if self.niters % Player.Q_UPDATE_FREQUENCY == 0:
                self.saveWeights()
                
            if log:
                self.training_log()
        
        
    def initModel(self):
        self.input_shape = (Player.BANDWIDTH, *Player.INPUT_SHAPE)
        
        if self.load_model and os.path.isfile(self.model_path):
            print("Loading model...")
            with open(self.model_path, 'r') as serialized:
                self.model = model_from_json(serialized.read().strip())

            assert self.input_shape == self.model.layers[0].input_shape[1:]

            try:
                self.model.load_weights(self.weights_path)
                print("Pretained model loaded")
            except:
                print("Model loaded")
            self.Qfunc = next(layer for layer in self.model.layers 
                              if isinstance(layer, Sequential))                            
        else:
            print("Initializing model...")
            self.Qfunc, self.model = getModel(self.input_shape)

        self.model.compile(loss='mse', optimizer=RMSprop(lr=0.00001))
       
        with open(self.model_path, '+w') as serialized:
            serialized.write(self.model.to_json())

        self.saveWeights()
        

    def loadTargetQ(self):
        with open(self.model_path, 'r') as serialized:
            model = model_from_json(serialized.read().strip())
            
        model.load_weights(self.weights_path)
        self.Qtarget = next(layer for layer in model.layers 
                            if isinstance(layer, Sequential))
        self.cacheQ = {}


    def saveWeights(self):
        self.model.save_weights(self.weights_path)
        self.loadTargetQ()
        print("Weights saved to {0}; target Q updated".format(self.weights_path))


    def training_log(self, log_path="log/player_log.txt"):
        with open(log_path, "+a") as log:
            log.write("{0} {1:.4f} {2:.4f}\n".format(
                self.niters, 
                self.avgQ,
                self.loss))
