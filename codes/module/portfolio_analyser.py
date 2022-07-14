from typing import Tuple
import pandas as pd
from datetime import datetime
import numpy as np

def get_sharpe_ratio(
    portfolio_return:pd.DataFrame,
    risk_free_rate: pd.DataFrame
    ) -> float:
    '''
    
    '''
    idx = portfolio_return.index
    rf_subset = risk_free_rate.loc[idx]
    # sr = np.mean(portfolio_performance.values- rf_subset.values) \
    #      / np.std(portfolio_performance.values - rf_subset.values)
    sr = np.mean(portfolio_return.values) / np.std(portfolio_return.values)

    return sr

def get_turnover(
    rebalance_option: str,
    stock_return_df: pd.DataFrame,
    portfolio_matrix: np.array,
    rolling_period:int
    ) -> Tuple[float, np.array]:
    '''
    

    '''
    stock_return_df += 1 #convert stock return to gross asset return

    if rebalance_option == 'month':
        rebalance_option = pd.Grouper(freq="M")
    elif rebalance_option == 'year':
        rebalance_option = pd.Grouper(freq="Y")
    else:
        raise Exception('rebalance option can be either \'month\' or \'year\'')

    stock_ret_rebalance = [group.values.tolist()[0] for name, group in stock_return_df.groupby(rebalance_option)] # unsafe expression using [0]
    stock_ret_rebalance = np.array(stock_ret_rebalance[rolling_period:], dtype=object)

    weighted_weights_matrix = portfolio_matrix * stock_ret_rebalance 
    weighted_weights_sum_vector = np.sum(weighted_weights_matrix, axis=1).reshape(-1,1)
    weighted_weights_diff_matrix = np.abs(portfolio_matrix[1:] - (weighted_weights_matrix/weighted_weights_sum_vector)[:-1])
    turnover_vector = np.sum(weighted_weights_diff_matrix, axis=1)
    turnover_avg = np.mean(turnover_vector)
    
    return turnover_avg, turnover_vector

def get_portfolio_net_return(
    portfolio_return:pd.DataFrame,
    turnover_vector:np.array,
    c: float
    ) -> pd.DataFrame:

    '''
    Calculate portfolio returns net of proportional transaction costs.
    
    '''

    port_ret_vec = portfolio_return.values.reshape(-1)
    port_net_ret_vec = (1 + port_ret_vec[1:]) * (1 - c * turnover_vector) - 1
    port_net_ret_vec = np.append(port_ret_vec[0], port_net_ret_vec)
    port_net_ret_df = pd.DataFrame(port_net_ret_vec, index=portfolio_return.index, columns=['portfolio net return'])

    return port_net_ret_df



