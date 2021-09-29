import sys
sys.path.append('/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/codes/')
from module.backtesting import *

import nonlinshrink as nls

class GMVP_backtesting(vectorized_backtesting):
    def get_portfolio(self, year):
        portfolio = self.get_GMVP(volatility_vector = self.volatility_aggregate.loc[year - 1], 
                            correlation_matrix = self.correlation_aggregate.loc[year - 1])
        return portfolio

class Linear_shrink_ledoit(vectorized_backtesting):
    '''
    Linear shrinkage method(ledoit) that shrink to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = LedoitWolf().fit(self.stocks_returns_aggregate.loc[year - 1]).covariance_
        portfolio = self.get_GMVP(covariance_matrix=  covariance_shrunk)
        return portfolio

class Linear_shrink_ledoit(vectorized_backtesting):
    '''
    Linear shrinkage method(ledoit) that shrink to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = LedoitWolf().fit(self.stocks_returns_aggregate.loc[year - 1]).covariance_
        portfolio = self.get_GMVP(covariance_matrix=  covariance_shrunk)
        return portfolio


class Nonlinear_shrink_ledoit(vectorized_backtesting):
    '''
    Non-linear shrinkage method(ledoit) that shrink to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = nls.shrink_cov(self.stocks_returns_aggregate.loc[year - 1])
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Linear_shrink_tbn(vectorized_backtesting):
    '''
    Linear shrinkage method(ledoit) that shrink sample covariance to Text Based Network
    '''
    def get_portfolio(self, year):
        alpha = LedoitWolf().fit(self.stocks_returns_aggregate.loc[year - 1]).shrinkage_
        covariance_shrunk = self.get_shrank_cov(correlation_matrix=self.correlation_aggregate.loc[year - 1].values,\
                                                shrink_target=self.tbn_combined.loc[year - 1].values,\
                                                volatility_vector=self.volatility_aggregate.loc[year - 1].values,
                                                a=alpha)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_50(vectorized_backtesting):
    '''
    Shrink sample covariance matrix 50 percent to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(covariance_matrix=self.covariance_aggregate.loc[year - 1].values,\
                                                shrink_target=np.identity(len(self.company_subset_PERMNO_vector)),\
                                                a=0.5)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_0(vectorized_backtesting):
    '''
    Shrink sample covariance matrix 0 percent to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(covariance_matrix=self.covariance_aggregate.loc[year - 1].values,\
                                                shrink_target=np.identity(len(self.company_subset_PERMNO_vector)),\
                                                a=0)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_100(vectorized_backtesting):
    '''
    Shrink sample covariance matrix 100 percent to identity matrix
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(covariance_matrix=self.covariance_aggregate.loc[year - 1].values,\
                                                shrink_target=np.identity(len(self.company_subset_PERMNO_vector)),\
                                                a=1)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_50_tbn(vectorized_backtesting):
    '''
    Shrink sample correlation matrix 50 percent to Text Based Network
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(correlation_matrix=self.correlation_aggregate.loc[year - 1].values,\
                                                shrink_target=self.tbn_combined.loc[year - 1].values,\
                                                volatility_vector=self.volatility_aggregate.loc[year - 1].values,
                                                a=0.5)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_0_tbn(vectorized_backtesting):
    '''
    Shrink sample correlation matrix 0 percent to Text Based Network
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(correlation_matrix=self.correlation_aggregate.loc[year - 1].values,\
                                                shrink_target=self.tbn_combined.loc[year - 1].values,\
                                                volatility_vector=self.volatility_aggregate.loc[year - 1].values,
                                                a=0)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class Shrink_100_tbn(vectorized_backtesting):
    '''
    Shrink sample correlation matrix 100 percent to Text Based Network
    '''
    def get_portfolio(self, year):
        covariance_shrunk = self.get_shrank_cov(correlation_matrix=self.correlation_aggregate.loc[year - 1].values,\
                                                shrink_target=self.tbn_combined.loc[year - 1].values,\
                                                volatility_vector=self.volatility_aggregate.loc[year - 1].values,
                                                a=1)
        portfolio = self.get_GMVP(covariance_matrix = covariance_shrunk)
        return portfolio

class MST_stock(vectorized_backtesting):
    '''
    Perform MST on stock returns' correlation
    '''
    def get_portfolio(self, year):
        get_distance_i_j = lambda rho, p = 2: np.sqrt(1 - abs(rho ** p))
        distance_matrix_stock = self.correlation_aggregate.loc[year - 1].apply(get_distance_i_j, args=[2])
        MST_stock = minimum_spanning_tree(distance_matrix_stock)
        MST_stock = MST_stock.toarray().astype(float)
        MST_stock = np.maximum(MST_stock, MST_stock.transpose())
        np.fill_diagonal(MST_stock, 1)
        portfolio = self.get_GMVP( volatility_vector = self.volatility_aggregate.loc[year - 1], 
                            correlation_matrix = MST_stock)
        return portfolio

class MST_tbn(vectorized_backtesting):
    '''
    Perform MST on Text Based Network
    '''
    def get_portfolio(self, year):
        get_distance_i_j = lambda rho, p = 2: np.sqrt(1 - abs(rho ** p))
        distance_matrix = self.tbn_combined.loc[year - 1].apply(get_distance_i_j, args=[2])
        MST = minimum_spanning_tree(distance_matrix)
        MST = MST.toarray().astype(float)
        MST = np.maximum(MST, MST.transpose())
        np.fill_diagonal(MST, 1)
        portfolio = self.get_GMVP( volatility_vector = self.volatility_aggregate.loc[year - 1], 
                            correlation_matrix = MST)
        return portfolio