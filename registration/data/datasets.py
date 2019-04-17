import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd


class MNIST(Dataset):
    """ MNIST dataset generated from the mnist csv files found on kaggle
    at [MNIST csv](https://www.kaggle.com/oddrationale/mnist-in-csv)
    """
    def __init__(self, path, size, transform=None):
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

    def build_dataset(self, path, size):
        """ Builds a dataset on initialization, random pairing done each time """
        ds = pd.read_csv(path, index_col=False)
        grouped_ds = ds.groupby(['label'])
        groups = grouped_ds.groups

        generated_ds = []
        for _ in range(size):
            label = random.randint(0, 9)
            pair = list(np.random.choice(groups[label], 2))
            generated_ds.append(pair)

        return ds, generated_ds

    def convert_pair_to_image(self, pair):
        image1_idx, image2_idx = pair
        image1 = self.mnist.iloc[image1_idx][1:]
        image2 = self.mnist.iloc[image2_idx][1:]
        image1 = np.array(image1).reshape((28,28)).astype(np.uint8)
        image2 = np.array(image2).reshape((28,28)).astype(np.uint8)
        return (image1, image2)

    def save_dataset(self):
        # write to npy file
        pass
