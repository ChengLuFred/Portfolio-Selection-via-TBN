import pandas as pd
#import pandas_datareader.data as web
from datetime import datetime
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import glob


class market_envrionment(object):

    def __init__(self, symbols_list, start_date, end_date):

        # initialization
        self.symols_list = symbols_list
        self.dates_range = pd.date_range(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date
        self.date = start_date
        self.date_index = 0        
        self.init_wealth = 10000
        self.reward_type = 'excess_return'
        # self.cost_rate = 0.01

        # get data
        self.load_data()

        # log
        self.states = []
        self.actions = []
        self.rewards = []
        self.log = {'state' : self.states, 'action': self.actions, 'reward' : self.rewards}

        # calculation variable
        '''
        TO DO
        '''

    def load_data(self):
        '''
        No return data, store stock, market and TBN data into object
        '''
        #file_path = os.getcwd() + '/data/'
        file_path = '../data/'
        file_type = '*.csv'
        file_list = glob.glob(file_path + file_type)
        # load TBN data
        idx = pd.Index
        tbn_combined = pd.DataFrame()
        for file in file_list:
            year = int(file.split('/')[-1][0:4])
            tbn = pd.read_csv(file,  header = 0, index_col = [0], engine='c')
            row_num = tbn.shape[0]
            year_idx = idx(np.repeat(year, row_num))
            tbn.set_index(year_idx, append = True, inplace = True)
            tbn_combined = tbn_combined.append(tbn)
        tbn_combined = tbn_combined.reorder_levels(order=[1,0])
        

        # load stock data
        file_name = 'stock_data.csv'
        self.stock_data = pd.read_csv(file_path + file_name,  header=0, index_col=[0], engine='c')

        # load market data
        file_name = 'F-F_Research_Data_Factors_daily.csv'
        self.interest_rate = pd.read_csv(file_name,  header=0, usecols=[0, 4], index_col=[0], engine='c').dropna()
        date_format = '%Y%m%d' # Y for year, m for month, d for day
        rf_date = pd.Index([datetime.strptime(str(x), date_format) for x in self.interest_rate.index])
        self.interest_rate.index = rf_date

    def get_state(self, action):
        if self.date in self.dates_range:
            state = self.portfolio_return()
            return state
        else:
            exit()

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
        state = self.get_state(action)
        reward = self.get_reward()
        self.record_log()
        return state, reward

    def record_log(self):
        """
        Save recording data to csv file
        """
        file_name = os.getcwd() + "/training/" + "training_data_%s.csv" % datetime.now().strftime("%D-%H:%M:%S")
        df = pd.DataFrame(self.log, columns = columns_name)
        df.to_csv(file_name, index = False)
        print("Output data to", file_name)

    def portfolio_return(self):
        '''
        TO DO
        '''

    def excess_return(self):
        '''
        TO DO

        '''

    def log_return(self):
        '''
        TO DO

        '''
    

    def sharpe_ratio(self):
        '''
        TO DO

        '''

    def moving_average(self):
        '''
        TO DO

        '''

    

    




    
        