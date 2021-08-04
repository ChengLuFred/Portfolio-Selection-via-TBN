# from module.packages import *
# basic
import pandas as pd
from datetime import datetime
import numpy as np

# operation system
import os
import glob
from tqdm import tqdm

# math
from numpy.linalg import inv
import math
from math import floor


class market_envrionment(object):

    def __init__(self):

        # get data
        self.load_data()
        
        # initialization
        #self.symols_list = symbols_list # unused for now
        #self.dates_range = pd.date_range(start_date, end_date)
        #self.start_date = start_date
        #self.end_date = end_date
        #self.date = start_date
        self.date_index_range = np.arange(1996, 2018) # unsafe expression
        self.date_index = self.date_index_range[0]

        # HYPER-PARAMETER
        self.observation_space = (1, ) # (state dimension) decided by user
        # self.action_space = np.array([0, 0.5, 1]) # action_space.shape[0] to get number of action
        self.alpha_step = 0.1
        self.action_space = np.arange(0, 1 + self.alpha_step, self.alpha_step)
        self.reward_type = 'log_return'
        # self.cost_rate = 0.01

        # log
        self.states = []
        self.actions = []
        self.rewards = []
        self.log = {'state' : self.states, 'action': self.actions, 'reward' : self.rewards}

        # calculation variable
        self.portfolio_return = 0

    def reset(self):
        # self.done = False
        self.portfolio_return = 0
        self.date_index = self.date_index_range[0]

        # get begning state
        stock_return = self.stock_returns.loc[1996] # for first year
        n = stock_return.shape[1] # number of symbols
        w = np.array([1/n] * n) # equal weights
        daily_return = stock_return @ w
        cumulative_return = (daily_return + 1).prod() - 1

        return cumulative_return

    def load_data(self):
        '''load local file(stock, market and TBN) under data folder into object
        Return:

            None

        Function:
            
            modify  self.tbn_combined
                    self.stock_prices
                    self.stock_returns
                    self.interest_rate
                    self.stock_correlation_aggregate
                    self.stock_volatility_aggregate
                    directly
        '''
        #file_path = os.getcwd() + '/data/'
        file_path = '../data/'
        file_type = 'TBN_*.csv'
        file_list = glob.glob(file_path + file_type)

        # load TBN data
        idx = pd.Index
        self.tbn_combined = pd.DataFrame()
        for file in file_list:
            year = int(file.split('/')[-1][4:8]) # not safe expression
            tbn = pd.read_csv(file,  header = 0, index_col = [0], engine='c')
            row_num = tbn.shape[0]
            year_idx = idx(np.repeat(year, row_num))
            tbn.set_index(year_idx, append = True, inplace = True)
            self.tbn_combined = self.tbn_combined.append(tbn)
        self.tbn_combined = self.tbn_combined.reorder_levels(order=[1,0])

        # load stock data
        file_name = 'stock_data.csv'
        self.stock_prices = pd.read_csv(file_path + file_name,  header=0, index_col=[0], engine='c')
        self.stock_prices = self.stock_prices.dropna(axis='columns') # drop incomplete data to form 26 columns

        # set year as index 
        date_format = '%Y-%m-%d' # Y for year, m for month, d for day
        stock_date = pd.Index([datetime.strptime(x, date_format) for x in self.stock_prices.index])
        self.stock_prices.index = [x.year for x in stock_date]

        # calculate stock return
        self.stock_returns = self.stock_prices.pct_change().dropna(axis='rows')

        # get correlation of stock return
        self.stock_correlation_aggregate = self.stock_returns.groupby(level=0).corr()
        self.stock_volatility_aggregate = self.stock_returns.groupby(level=0).std()

        # load market data
        file_name = 'F-F_Research_Data_Factors_daily.csv'
        self.interest_rate = pd.read_csv(file_path + file_name,  header=0, usecols=[0, 4], index_col=[0], engine='c').dropna()
        date_format = '%Y%m%d' # Y for year, m for month, d for day
        rf_date = pd.Index([datetime.strptime(str(x), date_format) for x in self.interest_rate.index])
        self.interest_rate.index = rf_date

    def get_state(self, action):
        '''Take agent's action and get back env's next state
        Args:
            action: a number (shrinkage intensity)
        Return:
            state - according to state mapping
        '''

        if not self.done():
            GMVP = self.get_GMVP(action)
            self.portfolio_return = self.get_portfolio_return(GMVP)
            state = self.portfolio_return
            return state
        else:
            print('The end of period\n')
            # exit()
    
    def done(self):
        '''Check whether goes to the end'''
        if self.date_index != self.date_index_range[-1]:
            return False
        else: 
            return True

    def get_reward(self):
        # map the reward_type to the reward function
        options = {'excess_return' : self.excess_return,
                   'log_return' : self.log_return,
                   'sharpe_ratio' : self.sharpe_ratio,
                   'moving_average' : self.moving_average
                  }
        
        reward_current = options[self.reward_type]()# whether self?
        return reward_current
    
    def step(self, action):
        '''Take a step in the environment

        '''
        
        # respond to agent
        state = self.get_state(self.action_space[action])
        reward = self.get_reward()

        # record experience
        # self.record_log()

        # move to next period
        self.date_index += 1 # unsafe expression

        return state, reward, self.done()

    def record_log(self):
        """
        Save recording data to csv file
        """
        file_name = os.getcwd() + "/training/" + "training_data_%s.csv" % datetime.now().strftime("%D-%H:%M:%S")
        df = pd.DataFrame(self.log)
        df.to_csv(file_name, index = False)
        print("Output data to", file_name)
    
    def get_GMVP(self, alpha):
        '''GMV portfolio as a function of intensity a
        Args:
            alpha is the shrinkage intensity (a number) which is agent's action
        Return:
            a column vector represting GMVP (np.array)
        '''
        # initialization
        a = alpha
        period_index = self.date_index # be careful
        R_1 = self.tbn_combined.loc[period_index].values
        R_2 = self.stock_correlation_aggregate.loc[period_index].values
        volatility_vector = self.stock_volatility_aggregate.loc[period_index]
        D = np.diag(volatility_vector) # diagnoal matrix with volatility on diagnoal
        one = np.ones(D.shape[0])

        # pre calculation
        R_3 = (1 - a) * R_1 + a * R_2 # new shrank correlated matrix
        H = D @ R_3 @ D # new shrank covariance matrix
        H_inv = inv(H)
        numerator = H_inv @ one
        denominator = one.T @ H_inv @ one

        # GMV porfolio
        x = numerator / denominator

        #return pd.DataFrame(H)
        return x.reshape((len(x), 1))

    def get_portfolio_return(self, portfolio_weights):
        '''calculate the portfolio return for next period
        Args:
            portfolio_weights (GMVP)

        Return:
            portfolio return (a number between 0 and 1)
        '''
        # initialization
        w = portfolio_weights
        period_index = self.date_index
        stocks_returns = self.stock_returns.loc[period_index + 1].values

        # portfolio return
        daily_return = stocks_returns @ w
        cumulative_return = (daily_return + 1).prod() - 1

        return(cumulative_return)

    def excess_return(self):
        '''
        TO DO

        '''

    def log_return(self):
        '''
        Calculate portfolio log return 
        '''
        R = self.portfolio_return
        r = np.log(R + 1)
        return r

    def sharpe_ratio(self):
        '''
        TO DO

        '''

    def moving_average(self):
        '''
        TO DO

        '''

    

    




    
        