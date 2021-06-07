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
from agent_network import *
from environment import market_envrionment
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# visulization
import igraph
import cairocffi
# import cairo
import matplotlib.pyplot as plt

# 
from scipy.sparse.csgraph import minimum_spanning_tree

def load_data():
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


def sharpe_ratio(portfolio_return, interest_rate):
    '''Given portfolio daily returns to calculate Sharpe ratio

    Argument:
        portfolio_return: a dataframe(T x 1) containing portfolio daily return time series
        interest_rate:    a dataframe(T x 1) containing risk free rate
    Return:
        a DataFrame (T x 1) of Sharpe ratio
    '''
    # initialization
    date_idx = portfolio_return.index
    r_f = interest_rate.loc[date_idx].values.flatten()

    # calculate SR
    excess_return = portfolio_return - r_f * 0.01
    expected_return = excess_return.mean() * 252 # annualized
    volatility = excess_return.std()* np.sqrt(252) # annualized
    sharpe_ratio = expected_return / volatility
    
    # transform
    temp = pd.DataFrame(sharpe_ratio, index=['Sharpe ratio']).T
    
    return temp