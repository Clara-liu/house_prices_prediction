import pandas as pd
import numpy as np

from torch.utils.data import Dataset


def split_data(file_path: str, test_series: dict) -> np.ndarray:
    """split data into train and test sets 

    :param file_path: path to the data file
    :type file_path: str
    :param test_series: information on what to reserve as test set e.g. {'Ealing':['ukhpi_averagePriceFlatMaisonette', ...], ...}
    :type test_series: dict
    :return: train and test total sequences i.e. not split into src, trg and trg_y
    :rtype: np.ndarray
    """
    df = pd.read_csv(file_path, sep='\t')
    authorities = df['Region'].unique()
    for region in authorities:
        pass


class HousePriceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
        