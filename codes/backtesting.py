'''
Supporting module for Portfolio Management Project
By Cheng Lu
'''
# basic
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import glob
import os
from numpy.linalg import inv
from bidict import bidict

# user-defined
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# visulization
#import igraph
#import cairocffi
# import cairo
import matplotlib.pyplot as plt

# 
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import chain
import nonlinshrink as nls
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
from scipy.stats import moment
from scipy.sparse.csgraph import minimum_spanning_tree
'''
Module Body
'''
class vectorized_backtesting:
    def __init__(self):
        self.stock_price, \
        self.interest_rate, \
        self.tbn_combined,\
        self.company_key = self.load_data()

        self.stocks_returns_aggregate, \
        self.correlation_aggregate, \
        self.covariance_aggregate, \
        self.volatility_aggregate = self.stock_analyser(self.stock_price)

        self.year_start = None
        self.year_end = None
        self.portfolio_returns = None
        self.portfolio = []

    def get_portfolio_daily_return_one_period(self, year):
        stocks_returns = self.stocks_returns_aggregate.loc[year]
        portfolio = self.get_portfolio(year)
        self.portfolio.append(portfolio) # record
        portfolio_returns = stocks_returns @ portfolio
        portfolio_returns = portfolio_returns.values.tolist()
        portfolio_returns = list(chain.from_iterable(portfolio_returns))

        return portfolio_returns

    def get_portfolio_daily_return(self, start, end):
        self.year_start = start
        self.year_end = end
        year_range = range(start, end + 1)
        self.portfolio_returns = [self.get_portfolio_daily_return_one_period(year) for year in year_range]
        self.portfolio_returns = list(chain.from_iterable(self.portfolio_returns))
        
        return self.portfolio_returns

    def get_portfolio(self, year):
        pass

    def get_shrank_cov(self, shrink_target, a, correlation_matrix = None, volatility_vector = None, covariance_matrix = None):
        # initialization
        if type(covariance_matrix) == type(None):
            R_1 = correlation_matrix
            R_2 = shrink_target
            D = np.diag(volatility_vector)

            # cov calculation
            R_3 = (1 - a) * R_1 + a * R_2 # new shrank correlated matrix
            H = D @ R_3 @ D # new shrank covariance matrix
        else:
            R_1 = covariance_matrix
            R_2 = shrink_target

            # cov calculation
            H = (1 - a) * R_1 + a * R_2 # new shrank covariance matrix

        return H

    def load_data(self):
        '''Load 1. stock price 2. interest rate 3. TBN data 4. company key

        Argument:
            None

        Return:
            1. stock price 
            2. interest rate  data frame
            3. TBN data       data frame
            4. company key    dictionary
        '''
        # initialization
        file_path = '../data/'
        rf_date_format = '%Y%m%d' # Y for year, m for month, d for day
        stock_date_format = '%Y-%m-%d' # Y for year, m for month, d for day

        # load company key
        file_name = 'gvkey_ticker.csv'
        company_key = pd.read_csv(file_path + file_name,  header=0, usecols=[1, 2], index_col = [0], engine='c')
        company_key = company_key.loc[~company_key.index.duplicated(keep='first')] # remove duplicated ticker
        company_key = company_key.to_dict()['gvkey'] # convert to dictionary
        company_key = bidict(company_key) # convert it to bidirectional dictionary

        # load stock price
        file_name = 'stock_data.csv'
        stock_price = pd.read_csv(file_path + file_name,  header=0, index_col=[0], engine='c')
        stock_price = stock_price.dropna(axis='columns') # drop incomplete data to form 26 columns
        #stock_subset = stock_price.dropna(axis='columns').columns # drop stock has incomplete data
        #tickers_key = company_key.loc[stock_subset].gvkey.values
        stock_date = pd.Index([datetime.strptime(x, stock_date_format) for x in stock_price.index])
        stock_price.index = [x.year for x in stock_date] # set year as index 

        # load interest rate
        file_name = 'F-F_Research_Data_Factors_daily.csv'
        interest_rate = pd.read_csv(file_path + file_name,  header=0, usecols=[0, 4], index_col=[0], engine='c').dropna()
        rf_date = pd.Index([datetime.strptime(str(x), rf_date_format) for x in interest_rate.index]) # convert index to date object
        interest_rate.index = rf_date

        # load TBN data
        file_path = '../data/'
        file_type = 'TBN_*.csv'
        file_list = glob.glob(file_path + file_type)

        idx = pd.Index
        tbn_combined = pd.DataFrame()
        for file in file_list:
            
            tbn = pd.read_csv(file,  header = 0, index_col = [0], engine='c')
            # np.fill_diagonal(tbn.values, 0)

            # index tbn by year
            row_num = tbn.shape[0]
            year = int(file.split('/')[-1][4:8]) # not safe expression
            year_idx = idx(np.repeat(year, row_num))
            tbn.set_index(year_idx, append = True, inplace = True)

            # combine each year tbn
            tbn_combined = tbn_combined.append(tbn)
        tbn_combined = tbn_combined.reorder_levels(order=[1,0])

        return stock_price, interest_rate, tbn_combined, company_key

    # - - - - - - - - - - - - - - - - - - - - -

    def stock_analyser(self, stock_price):
        '''Analyze stock price to calculate indicators

        Arguments:
            stock_price: T x N data frame
                        N - stock numbers
                        T - observation numbers
                        index is year number
        Returns:
            1. stock_returns
            2. correlation_aggregate
            3. covariance_aggregate
            4. volatility_aggregate
        '''
        # stock returns
        stock_returns = stock_price.pct_change().dropna(axis='rows') # the first row is dropped

        # correlation matrix for each year
        correlation_aggregate = stock_returns.groupby(level=0).corr() 

        # annualized covariance matrix for each year
        covariance_aggregate = stock_returns.groupby(level=0).cov() * 252

        # annualized volatility vector for each year
        volatility_aggregate = stock_returns.groupby(level=0).std() * np.sqrt(252)

        return stock_returns, correlation_aggregate, covariance_aggregate, volatility_aggregate

    # - - - - - - - - - - - - - - - - - - - - -

    def get_GMVP(self, correlation_matrix = None, volatility_vector = None, covariance_matrix = None):
        '''Get Global Minimum Variance Portfolio
        Args:
            correlation_matrix: correlation matrix used to build GMVP
            volatility_vector: vector of stocks volatility
        Returns:
            GMVP(column vector)
        '''
        # initialization
        if type(correlation_matrix) != type(None):
            R = correlation_matrix
            D = np.diag(volatility_vector)
            H = D @ R @ D
        else:
            H = covariance_matrix

        # pre calculation
        one = np.ones(H.shape[0]) # vector of 1s
        H_inv = inv(H)
        numerator = H_inv @ one
        denominator = one.T @ H_inv @ one

        # GMV porfolio
        x = numerator / denominator

        # reshape to column vector
        return x.reshape((len(x), 1))

    # - - - - - - - - - - - - - - - - - - - - -

    def get_sharpe_ratio(self):
        '''Given portfolio daily returns to calculate Sharpe ratio

        Argument:
            portfolio_return: a dataframe(T x 1) containing portfolio daily return time series
            interest_rate:    a dataframe(T x 1) containing risk free rate
        Return:
            a DataFrame (T x 1) of Sharpe ratio
        '''
        # initialization
        start_date = str(self.year_start) + '-01-01'
        end_date = str(self.year_end) + '-12-31'
        r_f = self.interest_rate.loc[start_date: end_date].values.flatten()
        portfolio_returns = np.array(self.portfolio_returns)

        # calculate SR
        excess_return = portfolio_returns - r_f * 0.01
        expected_return = excess_return.mean() * 252 # annualized
        volatility = excess_return.std()* np.sqrt(252) # annualized
        sharpe_ratio = expected_return / volatility
        
        # transform to dataframe
        #temp = pd.DataFrame(sharpe_ratio, index=['Sharpe ratio']).T
        
        return sharpe_ratio

    # - - - - - - - - - - - - - - - - - - - - -
    
    def get_turn_over_for_each_period(self):
        def get_turn_over_for_one_period(port_1, port_2):
            sell_and_buy = np.abs(np.array(port_2) - np.array(port_1))
            turn_over = sell_and_buy.sum() / 2
            return turn_over

        before_balance = self.portfolio[:-1]
        after_balance = self.portfolio[1:]
        turn_over_list = [get_turn_over_for_one_period(port_1, port_2) \
                          for port_1, port_2 in zip(before_balance, after_balance)]

        return turn_over_list

    

    