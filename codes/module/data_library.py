import pandas as pd
from datetime import datetime
import numpy as np

class data_library:
    def __init__(self) -> None:
        pass

    def get_stock_returns(
        name: str,
        frequency: str,
        start: str,
        end: str
        ) -> pd.DataFrame:
        '''
        Load local stock returns csv file.

        Args:

        Returns:
        '''

        file_path_dict = {
            ('industry', 'monthly'): \
            '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/49_Industry_Portfolios_Monthly_Value_Weighted.csv',
            ('industry', 'daily'):\
            '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/49_Industry_Portfolios_Daily_Value_Weighted.csv',
            ('B/M', 'monthly'):\
            '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/25_Portfolios_Formed_on_Size_and_Book-to-Marke_Monthly_Value_Weighted.CSV',
            ('moment', 'monthly'):\
            '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/10_Portfolios_Formed_on_Momentum_Monthly_Value_Weighted.CSV'
        }

        date_parser_dict = {
            'monthly': lambda x: datetime.strptime(str(x), "%Y%m"),
            'daily': lambda x: datetime.strptime(str(x), "%Y%m%d")
        }

        file_path = file_path_dict[(name, frequency)]
        data = pd.read_csv(file_path, index_col=0, parse_dates=True, date_parser=date_parser_dict[frequency])

        data = data[start: end]
        data = data.dropna(axis=1)
        data = data / 100 # convert to percentage

        return data

    def get_risk_free_rates(
        frequency:str
        ) -> pd.DataFrame:

        '''
        
        '''
        file_path_dict = {
            'monthly': \
            '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/F-F_Research_Data_Factors_Monthly.csv',
            }

        date_parser_dict = {
            'monthly': lambda x: datetime.strptime(str(x), "%Y%m"),
            'daily': lambda x: datetime.strptime(str(x), "%Y%m%d")
        }

        file_path = file_path_dict[frequency]
        data = pd.read_csv(file_path, index_col=0, usecols=[0, 4], parse_dates=True, date_parser=date_parser_dict[frequency])
        data = data / 100 # convert to percentage

        return data