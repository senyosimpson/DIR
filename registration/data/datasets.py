import torch
from torch.utils.data import Dataset
import os
import h5py
import random
import nibabel
import numpy as np
import pandas as pd
from glob import glob
from collections import namedtuple
from registration.analyze import imagetools


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


class LPBA40_H5(Dataset):
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


class LPBA40(Dataset):
    def __init__(self, root, train_split=0.8, training=True, seed=None, transform=None):
        if seed:
            random.seed(seed)
        self.root = root
        self.training = training
        self.dataset = self.build_dataset(train_split)
        self.transform = transform
        self.image_shape = imagetools.get_image_shape(f'{self.root}/native.mri.hdr')

    def __getitem__(self, idx):
        pair = self.dataset[idx]
        fixed_subj_no = pair.fixed
        moving_subj_no = pair.moving

        fixed_path = f'{self.root}/{fixed_subj_no}.native.mri.img'
        moving_path = f'{self.root}/{moving_subj_no}.native.mri.img'

        fixed = imagetools.get_image(fixed_path, self.image_shape)
        moving = imagetools.get_image(moving_path, self.image_shape)

        if not self.training:
            fixed_mask_path = f'{self.root}/{fixed_subj_no}.native.brain.mask.img'
            moving_mask_path = f'{self.root}/{moving_subj_no}.native.brain.mask.img'
            fixed_mask = imagetools.get_image_mask(fixed_mask_path, self.image_shape)
            moving_mask = imagetools.get_image_mask(moving_mask_path, self.image_shape)
            images = (fixed, moving, fixed_mask, moving_mask)
            idx = random.randint(20, 101)
            sample = np.array([imagetools.slice_image(image, dim=1, idx=idx) for image in images])
            assert sample.shape == (4, 256, 256)
            return sample

        fixed = np.pad(fixed,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
        moving = np.pad(moving,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
        fixed = imagetools.resize_image(fixed, (256,256,256))
        moving = imagetools.resize_image(moving, (256,256,256))

        idx = random.randint(40, 231)
        fixed = imagetools.slice_image(fixed, dim=1, idx=idx)
        moving = imagetools.slice_image(moving, dim=1, idx=idx)
        sample = np.array([fixed, moving])
        assert sample.shape == (2, 256, 256)
        return sample
        

    def __len__(self):
        if self.training:
            return len(self.dataset) * (231-40)
        return len(self.dataset) * (101-20)
    
    def build_dataset(self, train_split):
        """ Builds a dataset on initialization

        Arguments:
            train_split (int) : value from 0 to 1 denoting percentage of dataset to use for training
        
        Returns:
            sample (namedtuple) : tuple contains the subject numbers paired as (fixed, moving)
        """
        dataset = []
        Pair = namedtuple('Pair', 'fixed moving')
        image_paths = glob(os.path.join(self.root, '*.img'))
        for image1_path in image_paths:
            pair_image_paths = filter(lambda x: x != image1_path, image_paths)
            for image2_path in pair_image_paths:
                fixed_subj = 'S%s' % os.path.basename(image1_path)[1:3]
                moving_subj = 'S%s' % os.path.basename(image2_path)[1:3]
                dataset.append(Pair(fixed=fixed_subj, moving=moving_subj))
        
        random.shuffle(dataset)
        size = int(len(dataset) * train_split)
        if self.training:
            return dataset[:size]
        return dataset[size:]
