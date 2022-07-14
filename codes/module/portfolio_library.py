import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.special import betainc

def get_equal_weighted_portfolio(return_df):
    n_stock = return_df.shape[1]
    portfolio = [1 / n_stock] * n_stock

    return portfolio
    
def get_gmv(return_df):
    return_matrix = return_df.values
    sigma = np.cov(return_matrix, rowvar= False) # not sure do we need 252
    one = np.ones((sigma.shape[0], 1)) 
    sigma_inv = inv(sigma)

    numerator = sigma_inv @ one
    denominator = one.T @ sigma_inv @ one
    gmv = numerator / denominator

    return gmv

def get_zero_invest_portfolio(return_df: pd.DataFrame) -> np.array:
    '''
    Calculate the zero investment portfolio using given period data.

    Args:
        return_df: stock returns dataframe

    Returns:
        w_z: portfolio weights
    '''
    return_matrix = return_df.values
    sigma = np.cov(return_matrix, rowvar= False)
    mu = np.mean(return_matrix, axis=0, keepdims=True).T
    sigma_inv = inv(sigma)
    I = np.ones((len(mu), 1))
    mu_gmv = I.T @ sigma_inv @ mu / (I.T @ sigma_inv @ I)
    w_z = sigma_inv @ (mu - I @ mu_gmv)

    return w_z

def get_naive_combine_portfolio(
    return_df: pd.DataFrame,
    gamma: int = 3) -> np.array:
    '''
    Calculate the combine portfolio where don't take into account the estimation risk.
    In this portfolio, it combines GMV and zero investiment portfolio according to given gamma.
    It doesn't optimize the combining ratio.

    Args:
        return_df: stock return matrix
        gamma: risk aversion coefficient

    Returns:
        w_p: portfolio weight 
    '''
    c = 1
    w_z = get_zero_invest_portfolio(return_df)
    w_g = get_gmv(return_df)
    w_p = w_g + (c / gamma) * w_z

    return w_p

def get_opt_combine_portfolio(
    return_df: pd.DataFrame,
    rolling_period: int = 120,
    gamma: int = 3) -> np.array:
    '''
    Calculate the optimal combine portfolio where don't take into account the estimation risk.
    In this portfolio, it combines GMV and zero investiment portfolio according to given gamma and intensity c.
    The c is the optimal combining ratio that that maximizes the ex- pected out-of-sample utility.

    Args:
        return_df: stock return matrix
        gamma: risk aversion coefficient

    Returns:
        w_p: portfolio weight 
    '''
    return_matrix = return_df.values
    h = rolling_period
    N = return_df.shape[1]
    sigma = np.cov(return_matrix, rowvar= False)
    mu = np.mean(return_matrix, axis=0, keepdims=True).T
    sigma_inv = inv(sigma)
    I = np.ones((N, 1))

    psi = mu.T @ sigma_inv @ mu - (I.T @ sigma_inv @ mu) ** 2 / (I.T @ sigma_inv @ I)
    psi_a = ((h - N - 1) * psi - (N - 1)) / h + \
    (2 * psi ** ((N - 1) / 2) * (1 + psi) ** (- (h - 2) / 2)) / (h * betainc((N - 1) / 2, (h - N + 1) / 2, psi / (1 + psi)))
    k = (h - N) * (h - N -3) / (h * (h - 2))

    c = (psi_a * k) / (psi_a + (N - 1) / h)
    w_z = get_zero_invest_portfolio(return_df)
    w_g = get_gmv(return_df)
    w_p = w_g + (c / gamma) * w_z

    return w_p