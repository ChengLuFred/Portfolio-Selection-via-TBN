# basic
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import glob
import os
from numpy.linalg import inv
from bidict import bidict

# user-defined
import sys
sys.path.append('/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/codes/')
from module import *
from module.agent_network import *
from module.environment import market_envrionment
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# visulization
# import igraph
# import cairo
import matplotlib.pyplot as plt

# 
from scipy.sparse.csgraph import minimum_spanning_tree

class data_handler:
        def __init__(self) -> None:
                self.PERMNO_to_gvkey_mapping = self.get_PERMNO_to_gvkey_mapping()
                self.company_PERMNO_vector = self.get_company_PERMNO()
                self.company_gvkey_vector = self.get_company_gvkey()
            
        def get_PERMNO_to_gvkey_mapping(self) -> dict:
                '''
                Get PERMNO (permanent security identification number assigned by CRSP to each security) to 
                gvkey (six-digit number key assigned to each company in the Capital IQ Compustat database) mapping

                returns:
                        PERMNO_to_gvkey_mapping: a mapping from PERMNO to gvkey
                '''
                
                PERMNO_gvkey_key_file_path = '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/gvkey_id.csv'
                key_df = pd.read_csv(PERMNO_gvkey_key_file_path, sep=",", header=0, engine='c')
                PERMNO_gvkey_pairs = [(PERMNO, gvkey) for PERMNO, gvkey in zip(key_df['PERMNO'], key_df['gvkey'])]
                PERMNO_gvkey_pairs_unique = set(PERMNO_gvkey_pairs)
                PERMNO_to_gvkey_mapping = {PERMNO:gvkey for PERMNO, gvkey in PERMNO_gvkey_pairs_unique}

                return PERMNO_to_gvkey_mapping

        def get_company_PERMNO(self) -> np.array:
                '''
                Load CRSP stock return dataset and get company subset's PERMNO key vector
                '''
                stock_returns_file_path = '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/Data/permno_ret.csv'
                stock_returns_df = pd.read_csv(stock_returns_file_path, sep=",", header=0, index_col=[0],engine='c')
                company_subset_PERMNO_vector = stock_returns_df.columns

                return company_subset_PERMNO_vector

        def get_company_gvkey(self) -> np.array:
                '''
                Map company subset's PERMNO key vector to their gvkey vector
                '''
                company_subset_gvkey_vector = [self.PERMNO_to_gvkey_mapping[int(PERMNO)] for PERMNO in self.company_PERMNO_vector]

                return company_subset_gvkey_vector

        def extract_TNIC_network_subset(self, 
                                        file_input_path:str,
                                        file_output_path:str,
                                        key_list:np.array) -> None:
                '''
                Extract companies' TBN score subset from TNIC(Text-based Network Industry Classification) txt file.
                (available in Hoberger's database: http://hobergphillips.tuck.dartmouth.edu/idata/tnic_all_data.zip)
                Convert TBN scores vector into TBN matrix before output into local path.

                Args:
                        key_list: company subset gvkey list
                '''

                # initialization
                #file_input_path = '/Users/cheng/Documents/Research Data/Text Base Network/new/tnic_all_data/'
                file_input_type = "*.txt"
                #file_output_path = '/Users/cheng/Google Drive/PhD/Research/Portfolio Selection via TBN/data/TBN/'
                file_output_type = '.csv'
                file_list = glob.glob(file_input_path + file_input_type)
                idx = pd.IndexSlice

                # extract TBN scores from txt files
                for file in tqdm(file_list):

                        # read data
                        stock_pair_tbn_score = pd.read_csv(file, sep="\t", header=0, usecols=[0, 1, 2], engine='c')
                        stock_pair_tbn_score = stock_pair_tbn_score.set_index(['gvkey1', 'gvkey2'])
                        
                        # extract selected data
                        stock_pair_tbn_score_subset = stock_pair_tbn_score.loc[idx[key_list, key_list], ] # a long vector

                        # convert vector to matrix
                        stock_pair_tbn_score_subset = stock_pair_tbn_score_subset.unstack() # convert vector to matrix
                        stock_pair_tbn_score_subset.columns.names = [None,None] # clear multi column index 
                        stock_pair_tbn_score_subset.index.names = [None] # clear index head name
                        stock_pair_tbn_score_subset.columns = stock_pair_tbn_score_subset.index 
                        np.fill_diagonal(stock_pair_tbn_score_subset.values, 1) # fill diagnoal with value 1
                        stock_pair_tbn_score_subset = stock_pair_tbn_score_subset.replace([np.NaN], 0)

                        # save data
                        if not os.path.exists(file_output_path):
                                os.makedirs(file_output_path)
                        
                        identifier =file.split('/')[-1][-8:-4]
                        stock_pair_tbn_score_subset.to_csv(file_output_path + identifier + file_output_type)

        def export_dataframe_to_latex_table(
                                        df: pd.DataFrame, 
                                        table_name: str,
                                        output_path: str = '/Users/cheng/Dropbox/Apps/Overleaf/Portfolio Selection via Text Based Network/table',
                                        caption: str = None,
                                        label:str = None
                                        ) -> str:
                                        
                output_file_path = output_path + '/' + table_name + '.tex'
                float_format = "%.3f"

                latex_table = df.to_latex(output_file_path, 
                                        float_format=float_format, 
                                        caption=caption, 
                                        label=label)

                return latex_table