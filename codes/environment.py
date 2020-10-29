from packages import *

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
        No need for now
        '''

    def load_data(self):
        '''
        Return:

            None

        Function:

            load local file(stock, market and TBN) under data folder into object
            
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
        GMVP = self.get_GMVP(action)
        if self.date in self.dates_range:
            state = self.portfolio_return(GMVP)
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
    
    def get_GMVP(self, action):
        '''
        GMV portfolio as a function of intensity a

        return:
            a column vector represting GMVP (np.array)
        '''
        # initialization
        a = action
        period_index = self.date.year # be careful
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

    def portfolio_return(self, portfolio_weights):
        '''
        TO DO
        '''
        # initialization
        w = portfolio_weights
        period_index = self.date.year
        stocks_returns = self.stock_data.loc[period_index + 1].values

        # portfolio return
        daily_return = stocks_returns @ w
        cumulative_return = daily_return.cumprod()

        return(cumulative_return)


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

    

    




    
        