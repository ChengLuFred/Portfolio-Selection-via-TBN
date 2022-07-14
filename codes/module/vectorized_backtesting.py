import pandas as pd
from datetime import datetime
import numpy as np
from module.data_library import data_library
from module.portfolio_generator import portfolio_generator
from module.portfolio_analyser import get_sharpe_ratio, get_turnover, get_portfolio_net_return

def vectorized_backtesting(
    stock_return_df: pd.DataFrame, 
    portfolo_strategy_matrix: np.array,
    rebalance_option: str, 
    rolling_period:int):

    '''
    Backtesting framework. Prepare the data and the portfolio weights for each rolling period. 

    Args:
        stock_return_df:
        portfolo_strategy_matrix:
        rebalance_option:
        rolling_period:

    Return:
        portfolio_return_df: the time series of portfolio return. index is the date.
    
    '''

    
    if rebalance_option == 'month':
        rebalance_option = pd.Grouper(freq="M")
    elif rebalance_option == 'year':
        rebalance_option = pd.Grouper(freq="Y")
    else:
        raise Exception('rebalance option can be either \'month\' or \'year\'')

    stock_ret_rebalance = [group.values for name, group in stock_return_df.groupby(rebalance_option)]
    stock_ret_rebalance = np.array(stock_ret_rebalance[rolling_period:], dtype=object)

    n_period = stock_ret_rebalance.shape[0]
    n_stock = stock_ret_rebalance[0].shape[1]

    if n_period != portfolo_strategy_matrix.shape[0]:
        raise Exception('Portfolio matrix period doesn\'t align with stock return period.\
                         Stock return breaks down into {} rebalancing periods.\
                         Portfolio matrix has {} periods.'.format(n_period, portfolo_strategy_matrix.shape[0]))
    if n_stock != portfolo_strategy_matrix.shape[1]:
        raise Exception('portfolio matrix stock num doesn\'t align with stock return dimension')

    portfolio_return = np.concatenate([stock_ret @ portfolio for stock_ret, portfolio in zip(stock_ret_rebalance, portfolo_strategy_matrix)]) # can be replaced by array multiplication (stock_ret_rebalance @ portfolo_strategy_matrix)
    #idx_init = stock_ret_rebalance[0].index[0]
    idx_list = stock_return_df[rolling_period:].index
    portfolio_return_df = pd.DataFrame(portfolio_return, index=idx_list, columns=['portfolio_return'])

    return portfolio_return_df


def get_portfolio_performance(
    data_frequncy:str = 'monthly',
    data_name:str = 'industry',
    start='1969-07-01', 
    end='2018-12-01',
    rebalance_frequncy:str = 'month',
    get_portfolio:object = None,
    rolling_period:int = 120
) -> float:
    '''

    '''
    # data
    lib = data_library
    ret = lib.get_stock_returns(data_name, data_frequncy, start=start, end=end)
    rf = lib.get_risk_free_rates(frequency=data_frequncy)

    # preprocess
    idx = ret.index
    rf_subset = rf.loc[idx]
    ret = ret - rf_subset.values

    # portfolio matrix
    portfolio_manager = portfolio_generator()
    portfolio_manager.get_portfolio = get_portfolio
    portfolio_matrix = portfolio_manager.get_portfolio_matrix(
        stock_return_df=ret,
        rebalance_option=rebalance_frequncy,
        rolling_period=rolling_period
    )

    # portfolio return
    portfolio_return = vectorized_backtesting(
                                stock_return_df=ret,
                                rebalance_option=rebalance_frequncy,
                                portfolo_strategy_matrix=portfolio_matrix,
                                rolling_period=rolling_period
                            )

    # analysis
    sr = get_sharpe_ratio(portfolio_return,rf)
    turnover, turnover_vector = get_turnover(
                            stock_return_df=ret,
                            portfolio_matrix=portfolio_matrix,
                            rebalance_option=rebalance_frequncy,
                            rolling_period=rolling_period
    )
    portfolio_net_return = get_portfolio_net_return(
                    portfolio_return = portfolio_return, 
                    turnover_vector = turnover_vector, 
                    c = 0.002)
    sr_net = get_sharpe_ratio(portfolio_net_return, rf)
    performance = pd.DataFrame([sr, turnover, sr_net], index=['Sharpe ratio', 'Turnover', 'Sharpe ratio net'], columns=[data_name])

    return performance
