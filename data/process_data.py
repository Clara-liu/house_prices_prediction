import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, file_path: str, test_start: str, test_end: str, m: int=5, n: int=30):
        """_summary_

        :param file_path: absolute path to data file
        :type file_path: str
        :param test_start: start date for test data e.g. '2020/06'
        :type test_start: str
        :param test_end: end date for test data e.g. '2023-06'
        :type test_end: str
        :param m: the number of future data points to predict, defaults to 5
        :type m: int, optional
        :param n: the number of past data points used to predict future, defaults to 30
        :type n: int, optional
        """
        self.df = pd.read_csv(file_path, sep='\t')
        self.df['ukhpi_refMonth'] = pd.to_datetime(self.df['ukhpi_refMonth'])
        self.m = m
        self.n = n
        self.test_start = test_start
        self.test_end = test_end
        self.train = None
        self.test = None

    def get_train(self):
        df_train = self.df[self.df['ukhpi_refMonth']<self.test_start]
        