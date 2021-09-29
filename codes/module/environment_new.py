import sys
sys.path.append('/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/codes/')
from module.backtesting import *
from gym.utils import seeding
import gym

class market_environment(vectorized_backtesting):
    def __init__ (self, year_range):
        super().__init__()
        stock_num = self.stock_price.shape[1]
        self.action_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(1, ), dtype=np.float32)
        self.action_space_discrete = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(low = -100, high = 100, shape=(1, stock_num + 1), dtype=np.float32)
        self.seed(1)
        self.action_wrapper()
        self.done = False
        self.action = None
        self.state = None
        self.reward = None
        self.info = None
        self.reward_type = 'sharpe_ratio'
        self.year_range = year_range
        self.year = year_range[0]
        
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        '''
        Reset environment status to initial point.
        Return the initial state.

        Returns:
            state: np.array
        '''
        self.year = self.year_range[0]
        self.done = False
        self.state = None
        self.action = None
        self.reward = None
        self.info = None

        action = self.action_space.sample()
        state, reward, done, info = self.step(action)

        return state


    def step (self, action):
        '''
        Core function in environment. Take action as input, and respond to agent.
        Args:
            action: np.array
                    shrinkage intensity
        Returns:
            state: np.array
            reward: float
            done: Bool
        '''
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")
        else:
            assert self.action_space.contains(action)
            #self.action = action
            self.state = self.get_state(action)
            self.reward = self.get_reward()
            self.done = self.is_done()
            self.year += 1
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
        return [self.state, self.reward, self.done, self.info]

    def get_state(self, action):
        '''
        Take agent's action and get back env's next state
        Args:
            action: a number (shrinkage intensity)
        Return:
            state - according to state mapping
        '''
        if not self.done:
            self.action = action
            portfolio_mean_return = self.get_portfolio_mean_return(self.year, self.year)
            # portfolio_SR = self.get_sharpe_ratio()
            # portfolio_TO = self.get_turn_over_for_each_period()
            stocks_returns = self.get_stock_mean_returns(self.year)
            state = np.append(stocks_returns, portfolio_mean_return).reshape(1,-1)
            return state
        else:
            print('The end of period\n')
            # exit()
    
    # def get_portfolio(self, year):
    #     covariance_shrunk = self.get_shrank_cov(covariance_matrix=self.covariance_aggregate.loc[year - 1].values,\
    #                                             shrink_target=np.identity(23),\
    #                                             a=self.action)
    #     portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
    #     return portfolio
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(correlation_matrix=self.correlation_aggregate.loc[year - 1].values,\
                                                shrink_target=self.tbn_combined.loc[year - 1].values,\
                                                volatility_vector=self.volatility_aggregate.loc[year - 1].values,
                                                a=self.action)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio
    
    def is_done(self):
        '''
        Check whether agent arrive to the endpoint of the epoc
        '''
        if self.year != self.year_range[-1]:
            self.done = False
        else:
            self.done = True
            
        return self.done
    
    def get_reward(self):
        '''
        map the reward_type to the reward function
        '''
        options = {#'excess_return' : self.excess_return,
                   #'log_return' : self.log_return,
                   #'moving_average' : self.moving_average,
                   'sharpe_ratio' : self.get_sharpe_ratio
                  }
        
        reward = options[self.reward_type]()# whether self?
        return reward
    
    def action_wrapper(self):
        '''
        Action mapping to discretize continuous action space。
        Map agent's action(integer) to shrinkage intensity
        '''
        self.alpha_step = 0.1
        self.ACTION_MAPPING = np.arange(0, 1 + self.alpha_step, self.alpha_step)
        self.ACTION_MAPPING = self.ACTION_MAPPING.reshape(len(self.ACTION_MAPPING), 1)

    def render (self, mode="human"):
        s = "position: {}  reward: {}  info: {}"
        print(s.format(self.state, self.reward, self.info))

    def close(self):
        pass