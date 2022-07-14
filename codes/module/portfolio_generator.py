import pandas as pd
from datetime import datetime
import numpy as np

class portfolio_generator:
    def __init__(self) -> None:
        pass
    def get_portfolio_matrix(
        self,
        stock_return_df: pd.DataFrame, 
        rebalance_option: 'str',
        rolling_period: int) -> np.array:
        '''
        Slice stock df for each rolling period. 
        Put each period's df togather to form a list.
        For each rolling period calculate portfolio.
        Return a matrix(np.array) containing portfolio for each period.
        Period refers to month or year.
        Thus n period means n months or n years.

        Args:
            stock_return_df
            rebalance_option
            rolling_period

        Return:
            portfolio_matrix: a N by M matrix. N is stocks numbers. M is the number of rolling periods.
        
        '''
        if rebalance_option == 'month':
            rebalance_option = pd.Grouper(freq="M")
        elif rebalance_option == 'year':
            rebalance_option = pd.Grouper(freq="Y")
        else:
            raise Exception('rebalance option can be either \'month\' or \'year\'')

        n_period = len(stock_return_df.groupby(rebalance_option).groups)
        period_range_idx = np.arange(0, n_period+1)
        period_df_list = [stock_return_df.iloc[period: (period + rolling_period)] for period in period_range_idx[:-rolling_period-1]] #suspicious
        portfolio_matrix = [self.get_portfolio(period_df) for period_df in period_df_list]
        portfolio_matrix = np.array(portfolio_matrix).reshape((len(period_df_list), -1))

        return portfolio_matrix
    
    def get_portfolio(self, period_df):
        raise Exception('Please override get_portfolio() function!')