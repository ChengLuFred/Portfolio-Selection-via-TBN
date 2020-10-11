#######################################################################
# The environment imitates the stock market interacting with agent    #
# The main body of this file is class StockMarket                     #
# You may test the environment by running the function main()         #
# For test using, you have to turn "verbose" on in "_init_"           #
#######################################################################

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class StockMarket(object):

    def __init__(self, symbols, start_date, end_date):
        
        """
        symbols: a list of symbols you want to trade, the data type neeed to be list, even if you have only one argument.
        start_date: the starting date in the formate following "datetime" as (2018,1,1)
        end_date: similar as start_date, both define the traning and testing time period
        """

        # Initialize basic information
        self.dates_range = pd.date_range(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date 
        symbols_origin = symbols[:] #avoid shallow copy
        symbols.append('interest_rates')
        symbols.append('vix')     
        symbols.append('spy')   
        self.dateIdx = 0        
        self.init_cash = 10000
        self.cost_rate = 0.01

        # get data
        prices_all = self.get_data(symbols)
        self.stock = symbols_origin[0]
        self.prices = prices_all[symbols_origin] #be careful to slecet trading stock instead of benchmark
        self.prices_SPY = prices_all['spy']
        self.prices_VIX = prices_all['vix']
        self.prices_interest_rate = prices_all['interest_rates']
        self.date = prices_all.index[0] #first day
        self.date_length = len(prices_all)        

        # prepare for calculation
        self.cumulative_return = 0
        self.cumulative_profit = 0
        self.window_size = 5
        self.stock_window = deque([])
        self.stock_return =0
        self.stock_previous_prices = 0
        self.stock_current_prices = 0
        self.current_price = self.prices.loc[self.date, self.stock]
        self.previous_price = 0
        self.mean = 0
        self.S = 0 #needed for the calculation of standard deviation
        self.benchmark_value = 0   

        # record data
        self.data_output = []
        self.action = []
        self.portfolio = {'cash': self.init_cash, 'volume': [], 'price': []}        
        self.port_val = self.portfolio_value()

        # for debuging controlling
        self.verbose = [True, False][1]
        self.record = [True, False][1]
        self.reward_type = ["log return", 
                            "sharpe ratio", 
                            "profit", 
                            "moving average", 
                            "exponential moving average",
                            "geometric moving average"][3] # change the num to control reward type

    def reset(self):
        self.dateIdx = 0
        self.date = self.prices.index[self.dateIdx]        
        self.init_cash = 10000
        self.cumulative_return = 0
        self.stock_window = deque([])
        self.cumulative_profit = 0
        self.stock_return =0
        self.stock_previous_prices = 0
        self.stock_current_prices = 0
        self.current_price = self.prices.loc[self.date, self.stock]
        self.previous_price = 0
        self.mean = 0
        self.S = 0
        self.benchmark_value = 0
        self.data_output = []
        self.action = []
        self.portfolio = {'cash': self.init_cash, 'volume': [], 'price': []}        
        self.port_val = self.portfolio_value()

    def get_data(self, symbols):
    	"""
    	get the stocks data from folder data
    	"""
    	cwd = os.getcwd() + ''
    	dir = cwd + "/data/"
    	df = pd.DataFrame()
    	for symbol in symbols:
            file_name = dir + symbol.lower() + '.csv'
            if not os.path.exists(file_name):
                data = pd.DataFrame(web.DataReader(symbol.upper(), 'yahoo', self.start_date, self.end_date))
                data.to_csv(file_name)
                print('downloading data', symbol.upper())                        
            data = pd.read_csv(file_name, index_col=["Date"], parse_dates=['Date'])
            print("Loading stock data", symbol)
            if 'Close' in data:
                df[symbol] = data['Close']
            else:
                df[symbol] = data['Rate']           	
    	return df[df.index.isin(self.dates_range)].sort_index() 

    def Sharpe_Ratio(self, R_p, r_f, n):
    	"""
    	Compute the Sharpe ratio incrementally
        Inspired by http://datagenetics.com/blog/november22017/index.html
    	"""
    	if n == 1:
    		self.mean = R_p - r_f
    		self.S = 0
    		return 0
    	else:
    		mean_0 = self.mean
    		S_0 = self.S
    		excess_return = R_p - r_f
    		mean_1 = mean_0 + (excess_return - mean_0) / n
    		S_1 = S_0 + (excess_return - mean_0) * (excess_return - mean_1)
    		standard_deviation = math.sqrt(S_1 / (n-1))
    		sharpe_ratio = (excess_return) / standard_deviation
    		self.mean = mean_1
    		self.S = S_1
    		return sharpe_ratio

    def step(self, action):
        """
        Take an action, and move the date forward,
        record the reward of the action and date
        action: buy, sell, hold
        return (reward, state(stock return))
        """ 
        # set trading volume as fixed       
        volume = self.init_cash / self.prices.loc[self.prices.index[0], self.stock]
        #volume = 100

        # get the previous day's position and record today's position
        if not self.action: 
            F_pre = 0
            if self.verbose: print("The first day")
        else:
            F_pre = self.action[-1]
        F = action
        self.action.append(F) # record today's action        

        # get stock price and calculate stock return
        self.previous_price = self.current_price
        self.current_price = self.prices.loc[self.date, self.stock]
        stock_return = (self.current_price - self.previous_price) / self.previous_price
        log_return = np.log(self.current_price / self.previous_price)

        # calculate the profit
        self.cumulative_profit += volume * (F_pre * (self.current_price - self.previous_price) - self.cost_rate * abs(F - F_pre))
        self.benchmark_value += self.benchmark()

        # state
        state = stock_return

        # reward
        if self.reward_type == "log return":
            reward = log_return * volume * F_pre
        elif self.reward_type == "sharpe ratio":
            reward = self.Sharpe_Ratio(portfolio_return, self.prices_interest_rate[self.date]*0.01, self.dateIdx+1) * volume * F_pre
        elif self.reward_type == "profit":
            reward = self.port_val - old_port_val
        elif self.reward_type == "exponential moving average":
            if len(self.stock_window) < self.window_size:
                self.stock_window.append(stock_return)
            else:
                self.stock_window.append(stock_return)
                self.stock_window.popleft()
            reward = pd.DataFrame(np.array(self.stock_window)).ewm(span=len(self.stock_window), adjust=False).mean() # an exponetial window
            reward = np.array(reward)[-1] * volume * F_pre
        elif self.reward_type == "moving average":
            if len(self.stock_window) < self.window_size:
                self.stock_window.append(stock_return)
            else:
                self.stock_window.append(stock_return)
                self.stock_window.popleft()
            reward = np.mean(self.stock_window) * volume * F_pre
        elif self.reward_type == "geometric moving average":
            if len(self.stock_window) < self.window_size:
                self.stock_window.append(stock_return)
            else:
                self.stock_window.append(stock_return)
                self.stock_window.popleft()
            reward = (1+np.array(self.stock_window)).prod()**(1.0/len(self.stock_window)) - 1
            reward = reward * volume * F_pre
        # record data for output
        self.data_output.append([self.date.isoformat()[0:10],
                                 str(self.prices.loc[self.date, self.stock]),
                                 action,
                                 str(self.cumulative_profit), 
                                 str(self.stock_return),
                                 str(self.benchmark_value)])
        #state = self.get_state(self.date)

        # debug using   
        if False:
            print(self.date, "action", action,
                             "reward %.3f" % reward,
                             "state %.3f" % state,
                             "cumulative profit %.3f" % self.cumulative_profit,
                             "log return", log_return,
                             "date index", self.dateIdx)
        # move forward
        self.dateIdx += 1
        if self.dateIdx < self.date_length:
            self.date = self.prices.index[self.dateIdx]
        else:
            if self.verbose: print("The next day is out of range")
        
        return reward, state


    def get_state(self, date):
        """
        return state of the market, i.e. prices of certain symbols,
        number of shares hold
        """
        if date not in self.dates_range:
            if self.verbose: print('Date was out of bounds.')
            if self.verbose: print(date)
            exit()

        #define the state as stock's daily return
        return self.stock_return

    def portfolio_value(self):
    	"""
    	calculate the portfolio's value for current state
    	"""
    	value = self.portfolio['cash']
    	value += (sum(self.portfolio['volume']) * self.prices.loc[self.date, self.stock])
        #if self.verbose: print(self.date,"portfolio value %.3f" % value)

    	return value

    def get_stock_return(self):
        """
        calculate the stock's daily return
        """
        self.stock_previous_prices = self.stock_current_prices
        self.stock_current_prices = self.prices.loc[self.date, self.stock]
        if not self.stock_previous_prices: # for the first day
            return 0, 0        
        self.stock_return = (self.stock_current_prices - self.stock_previous_prices)/self.stock_previous_prices
        log_return = np.log(self.stock_current_prices/self.stock_previous_prices)

        # for debug using
        if self.verbose:
            print(self.date, "current price %.3f" % self.stock_current_prices,
                             "previous price %.3f" % self.stock_previous_prices,
                             "stock return %.3f" % self.stock_return)            

        return self.stock_return, log_return

    def episode_end(self):
        """
        output the baseline value at the end
        """        
        if not (self.dateIdx < self.date_length) and self.verbose:
            print('\n\n\n*****')
            print('The end of time period')
            print('Portfolio Value', self.port_val)
            print('Benchmark ', self.benchmark())
            print('*****\n\n\n')

        return self.dateIdx < self.date_length

    def data_graph(self, name):
        """
        output profit and benchmark in a graph
        """
        date = [dt.datetime.strptime(i[0],'%Y-%m-%d') for i in self.data_output]

        cumulative_profit = [float(i[3]) for i in self.data_output]
        benchmark = [float(i[5]) for i in self.data_output]
        action = [float(i[2]) for i in self.data_output]
        x_label = [date[0], date[int(len(date)/3)], date[int(2*len(date)/3)], date[-1]]

        # plot cumulative profit
        plt.subplot(2, 1, 1)
        plt.plot(date, cumulative_profit, color='black')
        plt.plot(date, benchmark, color='green')
        plt.ylabel('Profit')
        plt.xlabel('Time Step')
        plt.xticks(x_label, fontsize=7, rotation=45)

        # plot action series
        plt.subplot(2, 1, 2)
        plt.plot(date, action, drawstyle='steps-post',color='black')
        plt.ylabel('Action')
        plt.xlabel('Time Step')
        plt.xticks(x_label, fontsize=7, rotation=45)
        plt.savefig('images/profit_action_'+name+'.png')
        plt.close()


    def output_to_file(self):
        """
        Save recording data to csv file
        """
        file_name = os.getcwd() + "/training/" + "training_data_%s.csv" % dt.datetime.now().strftime("%H-%M-%S")
        columns_name = ["Date", "Price", "Action", "Cumulative profit", "Stock Return", "Benchmark"]
        df = pd.DataFrame(self.data_output, columns = columns_name)
        df.to_csv(file_name, index = False)
        print("Output data to", file_name)

    def benchmark(self):
        """
        The benchmark is the spy's value
        """
        volume = self.init_cash / self.prices_SPY[0]
        #volume = 100
        if self.dateIdx == 0:
            current_price_base = 0
            previous_price_base = 0
        else:
            current_price_base = self.prices_SPY.loc[self.date]
            previous_day_base = self.prices_SPY.index[self.dateIdx - 1]
            previous_price_base = self.prices_SPY.loc[previous_day_base]
        profit_base = volume * (current_price_base - previous_price_base)
        return profit_base

def main():
    """
    Test the environment with some random actions
    """    
    sim = StockMarket(['peg'], dt.datetime(2014, 3, 1), dt.datetime(2014, 5, 1))
    pmf = [0.3, 0.4, 0.3]
    while sim.episode_end():
        action = np.random.choice(3, p=pmf) - 1
        reward, state = sim.step(action)
        #print(reward)
    #sim.output_to_file()
    sim.data_graph("random")

if __name__ == '__main__':
    main()
