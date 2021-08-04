'''
Adapted from Emily Liu's class note
'''
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import pandas as pd

from agent_network import *
from environment import market_envrionment

def main():

    # initialization for agent and environment
    env = market_envrionment()
    state_size = env.observation_space[0] # given from environment
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")

    # hyper-parameter
    done = False
    batch_size = 5
    history = []
    EPISODES = 1000
    
    for e in range(EPISODES): # one episode is M trading years in a period
        
        # initialize state
        state = env.reset()
        state = np.reshape(state, [1, state_size]) # an array containing only one array [[a,b,c,d]]
        rewards = 0
        
        for time in range(17): # how many years in a training period

            # take an action
            action = agent.act(state)
            
            # environment responds to the action and return new state and reward
            next_state, reward, done = env.step(action)
            
            # record reward
            rewards += reward
            
            # reshape state since the neural network expects an array with rank 2 (can be reshaped within env)
            next_state = np.reshape(next_state, [1, state_size]) # an 1 x n 2d array
            
            # record the experience for replay
            agent.memorize(state, action, reward, next_state, done) # record every trading 
            
            # transit to next state
            state = next_state
            
            # determine if the game is over or not
            if done:
                break
                
            # if there are enough experiences accumulated, replay to train the network    
            if len(agent.memory) > batch_size: # batch_size = 2 to make agent learn for every 3 trading events
                agent.replay(batch_size)
                
        print("episode: {}/{}, score: {}, e: {:.2}, rewards: {}"
                      .format(e, EPISODES, time, agent.epsilon, rewards))
        
        history.append([e, time, agent.epsilon, rewards])

    # save training plots
    training_plot(history)

def training_plot(history):
    df = pd.DataFrame(history, columns =["episode", "total_time","epsilon",'reward'])
    df.set_index("episode")
    df.to_csv('../output/RL_training_data.jpg')
    figs, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

    axs[0].set_title('Total Rewards for each episode')
    axs[0].plot(df.index, df['reward'].rolling(5).mean())
    axs[0].set_ylabel('total rewards')

    axs[1].set_title('Epsilon(exploring rate) for each episode')
    axs[1].plot(df.index, df['epsilon'])
    axs[1].set_ylabel('epsilon')

    plt.tight_layout()
    figs.savefig('../picture/RL_training_plots.jpg')
    plt.show()

if __name__ == '__main__':
    env_test = market_envrionment()