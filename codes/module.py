import numpy as np
import pandas as pd

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