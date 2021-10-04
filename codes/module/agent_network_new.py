 # basic
import random
import gym
import numpy as np
import pandas as pd
from collections import deque

# model
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
#tf.config.run_functions_eagerly(True)


class DQNAgent:
    '''Agent Class implemented using network

    '''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        '''action network: mapping input state to Q(s,a)
        Return:
            An network approximate Q(s,a)
        '''
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''take state as input for action network to predict next action
        Return:
            an integer(index of action)
        '''
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size, 1)[0]
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action with highest Q value

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) # an array of n element-array [[a,b]] n is dimension of action space
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.model.summary()
        
        
    def policy_gradient_loss(self, Returns):
        def modified_crossentropy(action,action_probs):
            cost = K.categorical_crossentropy(action, action_probs,\
                                              from_logits=False, axis=1)
            g = K.squeeze(Returns, axis = -1)
            cost = cost * g
            return K.mean(cost)
        return modified_crossentropy

    def _build_model(self):

        state = Input(shape = (self.state_size,), name = 'states')
        rewards = Input(shape = (1,), name = 'rewards')
        dense1 = Dense(24, activation='relu')(state)
        dense2 = Dense(24, activation='relu')(dense1)
        output = Dense(self.action_size, activation='softmax')(dense2)
        model = Model(inputs=[state, rewards], outputs=output)
        opt = Adam(learning_rate=self.learning_rate)
        policy_loss = self.policy_gradient_loss(rewards)
        model.compile(loss=policy_loss, optimizer=opt)
        
        return model

    def memorize(self, state, action, reward, next_state, done):
        '''
        
        '''
        action_encode = np.zeros([self.action_size])
        action_encode[action] = 1
        self.actions.append(action_encode)# one hot encoding
        self.states.append(state)
        self.rewards.append(reward)
    
    def clean_memory(self):
        self.states, self.probs, self.actions, self.rewards = [], [], [], []


    def act(self, state):
        
        prob = self.model.predict([state, np.zeros((1,1))]).flatten()
        self.probs.append(prob)
        action = np.random.choice(self.action_size, p=prob)
        return action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    

    def train(self):
        '''
        Train agent network. Reshape rewards and actions sequence. 
        Then update network on one epoch. Finally, clean up agent memory.

        Args:
            none
        Returns:
            none
        '''

        rewards = np.vstack(self.rewards) # Stack arrays in sequence vertically (row wise)
        # rewards = self.discount_rewards(rewards)
        # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

        X = np.stack(self.states, axis = 0) # convert list of array to array
        X = np.expand_dims(X, axis = -1) # expanded axes with new axis
        Y = np.vstack(self.actions)
        
        self.model.train_on_batch([X, rewards], Y)
        self.clean_memory()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)