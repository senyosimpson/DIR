import torch
from torch.utils.data import Dataset
import h5py
import random
import numpy as np
import pandas as pd


class MNIST(Dataset):
    """ MNIST dataset generated from the mnist csv files found on kaggle
    at [MNIST csv](https://www.kaggle.com/oddrationale/mnist-in-csv)
    """
    def __init__(self, path, size, seed=None, transform=None):
        if seed:
            random.seed(seed)
        self.mnist, self.mnist_pairs = self.build_dataset(path, size)
        self.transform = transform

    def __getitem__(self, idx):
        pair = self.mnist_pairs[idx]
        sample = self.convert_pair_to_image(pair)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.mnist)

    def build_dataset(self, path, size, save_path=None):
        """ Builds a dataset on initialization, random pairing done each time """
        ds = pd.read_csv(path, index_col=False)
        grouped_ds = ds.groupby(['label'])
        groups = grouped_ds.groups

        generated_ds = []
        for _ in range(size):
            label = random.randint(0, 9)
            pair = list(np.random.choice(groups[label], 2))
            generated_ds.append(pair)

        if save_path:
            # save generated dataset to csv
            pass
        return ds, generated_ds

    def convert_pair_to_image(self, pair):
        image1_idx, image2_idx = pair
        image1 = self.mnist.iloc[image1_idx][1:]
        image2 = self.mnist.iloc[image2_idx][1:]
        image1 = np.array(image1).reshape((28,28)).astype(np.uint8)
        image2 = np.array(image2).reshape((28,28)).astype(np.uint8)
        return (image1, image2)


class LPBA40(Dataset):
    def __init__(self, path, training=True, data_splits=(62592, 9984), transform=None):
        """
        Arguments:
            path (str) : path to h5py file
            data_splits (tuple) : split of the dataset specified as either absolute values
                or values adding up to 1 i.e (0.8, 0.2). format is (train_size, test_size)
        """
        self.dataset = h5py.File(path, 'r')['Brain MRI Dataset']
        self.train_split, self.test_split = self.get_splits(data_splits)
        self.transform = transform
        self.training = training


    def __getitem__(self, idx):
        if self.training:
            sample = self.dataset[idx]
        else:
            sample = self.dataset[-idx]

        if self.transform:
            sample = self.transform(sample)
        return sample
                

    def __len__(self):
        if self.training:
            return self.train_split
        else:
            return self.test_split
    
    def get_splits(self, data_splits):
        train_split, test_split = data_splits
        if train_split < 1:
            train_split = int(len(self.dataset) * train_split)
            test_split = int(len(self.dataset) * test_split)
            return train_split, test_split
        return train_split, test_split