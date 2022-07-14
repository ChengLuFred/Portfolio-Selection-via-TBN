import sys
sys.path.append('/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/codes/')
import warnings
warnings.filterwarnings("ignore")
from module.backtesting import *
from module.environment_with_new_bt import *
from module.agent_network_new import *
import pandas as pd
import numpy as np
from tqdm import tqdm

def is_converge(data, threshold=0.05, window_size=5):
    converge = False
    i = len(data)
    mean_reward_current = np.mean(data[i-window_size:i])
    mean_reward_previous = np.mean(data[i-2*window_size:i-window_size])
    # print(mean_reward_current)
    # print(mean_reward_previous)
    if abs(mean_reward_current - mean_reward_previous) < threshold:
        print('Converged on episode {}'.format(i))
        converge = True
    return converge

def training(year_range):
    # initialize environment and agent
    env = market_environment(year_range)
    ACTION_MAPPING = env.ACTION_MAPPING
    state_size = env.observation_space.shape[1] # given from environment
    action_size = len(ACTION_MAPPING)
    agent = DQNAgent(state_size, action_size)
    EPISODE_RANGE = 1000
    batch_size = 6
    converge = False
    window_size = 50
    history = {'episode': [], "epsilon": [],'rewards': []}
    path = '../log/'
    model_name = 'DQN_'

    for e in tqdm(range(EPISODE_RANGE)):
        
        # initialize state
        state = env.reset()
        rewards = 0

        for time_window in year_range: # how many years in a training period

            # take an action
            action = agent.act(state) 
            
            # environment responds to the action and return new state and reward
            next_state, reward, done, info = env.step(ACTION_MAPPING[action])
            
            # record reward
            rewards += reward
            
            # record the experience for replay
            agent.memorize(state, action, reward, next_state, done) # record every trading 
            
            # transit to next state
            state = next_state

                # determine if the training is over or not
            if done:
                break

        # replay to train the network    
        if len(agent.memory) > batch_size: # batch_size = 2 to make agent learn for every 3 trading events
            agent.replay(batch_size)

        # log information
        history['episode'].append(e)
        history['epsilon'].append(agent.epsilon)
        history['rewards'].append(rewards)
        #history.append([e, agent.epsilon, rewards])

        # converge decision
        if e % window_size == 0 and e > window_size:
            converge = is_converge(history['rewards'], threshold=0.001, window_size=window_size)

        if converge:
            agent.save(path + model_name + str(year_range[-1] + 1))
            break
        
    return agent, history

def testing(testing_period, agent):
    # initialize environment and agent
    env = market_environment(testing_period)
    ACTION_MAPPING = env.ACTION_MAPPING
    agent.epsilon = 0
    performance = {'year':testing_period[-1],'action': [], 'reward': [], 'turnover': []}

    # initialize state
    state = env.reset()

    for _ in testing_period: # how many years in a testing period

        # take an action
        action = agent.act(state) 
        
        # environment responds to the action and return new state and reward
        next_state, reward, done, info = env.step(ACTION_MAPPING[action])

        # transit to next state
        state = next_state

        # determine if the testing is over or not
        if done:
            performance['action'] = ACTION_MAPPING[action]
            performance['reward'] = reward
            performance['turnover'] = env.get_turn_over_for_each_period()[-1] # get the turnover from last year(the next year)
            break

    return performance