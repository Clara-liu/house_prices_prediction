import logging

import pandas as pd
import numpy as np
from typing import Tuple, Callable

from torch.utils.data import Dataset
from torch import from_numpy, Tensor


def split_data(file_path: str, test_series: list, total_seq_len: int = 340) -> Tuple[np.ndarray, np.ndarray, list]:
    """split data into train and test sets 

    :param file_path: path to the data file
    :type file_path: str
    :param test_series: information on what to reserve as test set e.g. ['Ealing', 'Greenwhich', ...]
    :type test_series: list
    :return: train and test total sequences (not split into src, trg and trg_y) [n_seq, seq_len, n_feature] and region labels for test
    :rtype: tuple
    """
    df = pd.read_csv(file_path, sep='\t')
    authorities = df['Region'].unique()
    train = []
    test = []
    labels = []
    for region in authorities:
        sliced = df.loc[df['Region'] == region]
        if sliced.shape[0] != total_seq_len:
            logging.warning(f'Time series for region {region} does not match total length: {total_seq_len}. Will not use.')
        else:
            # get rid of date and region columns
            numeric_df = sliced.select_dtypes(include=np.number)
            # get region info for test sets
            add_region = lambda x: '_'.join([region, x])
            # transform slice to numpy
            numeric_data = [*numeric_df.to_numpy().T]
            if region in test_series:
                test = test + numeric_data
                # save region and property type information
                region_info = map(add_region, list(numeric_df.columns))
                labels = labels + list(region_info)

            else:
                train = train + numeric_data
    shape = lambda x: np.expand_dims(np.array(x),2)  # add feature dimension and convert to numpy 
    return shape(train), shape(test), labels


class HousePriceDataset(Dataset):
    def __init__(self, data: np.ndarray, transform: Callable = np.log) ->Tensor:
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """return data instance from dateset. [n_seq, seq_len, feature_dim] sequence is not split into src and trg!

        :param idx: idx to retrieve data
        :type idx: int
        :return: TRANSFORMED DATA according to transform function defined in init
        :rtype: Tensor
        """
        return from_numpy(self.transform(self.data[idx]))