import torch
from torch.utils.data import Dataset
import os
import h5py
import random
import nibabel
import skimage
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


class LPBA40(Dataset):
    def __init__(self, root, train_split=0.9, training=True, seed=None, transform=None):
        if seed:
            random.seed(seed)
        self.root = root
        self.training = training
        self.dataset = self.build_dataset(train_split)
        self.transform = transform
        self.image_shape = imagetools.get_image_shape(os.path.join(self.root, 'native.mri.hdr'))

    def __getitem__(self, idx):
        pair = self.dataset[idx]
        fixed_subj_no = pair.fixed
        moving_subj_no = pair.moving
        slice_depth = pair.depth

        fixed_path = os.path.join(self.root, '%s.native.mri.img' % fixed_subj_no)
        moving_path = os.path.join(self.root, '%s.native.mri.img' % moving_subj_no)

        fixed = imagetools.get_image(fixed_path, self.image_shape)
        moving = imagetools.get_image(moving_path, self.image_shape)

        if not self.training:
            fixed_mask_path = os.path.join(self.root, '%s.native.brain.mask.img' % fixed_subj_no)
            moving_mask_path = os.path.join(self.root, '%s.native.brain.mask.img' % moving_subj_no)
            fixed_mask = imagetools.get_image_mask(fixed_mask_path, self.image_shape)
            moving_mask = imagetools.get_image_mask(moving_mask_path, self.image_shape)
            images = [fixed, moving, fixed_mask, moving_mask]
            sample = np.array([imagetools.slice_image(image, dim=1, idx=slice_depth) for image in images])
            assert sample.shape == (4, 256, 256)
            if self.transform:
                sample = self.transform(sample)
            return sample

        fixed = imagetools.slice_image(fixed, dim=1, idx=slice_depth)
        moving = imagetools.slice_image(moving, dim=1, idx=slice_depth)

        sample = np.array([fixed, moving])
        assert sample.shape == (2, 256, 256)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)
    
    def build_dataset(self, train_split):
        """ Builds a dataset on initialization

        Arguments:
            train_split (int) : value from 0 to 1 denoting percentage of dataset to use for training
        
        Returns:
            dataset list(namedtuple) : list containing namedtuple objects of subject numbers
                paired as (fixed, moving) and slice index
        """
        dataset = []
        Pair = namedtuple('Pair', 'fixed moving depth')
        image_paths = glob(os.path.join(self.root, '*.img'))
        for image1_path in image_paths:
            pair_image_paths = filter(lambda x: x != image1_path, image_paths)
            for image2_path in pair_image_paths:
                fixed_subj = os.path.basename(image1_path)[:3]
                moving_subj = os.path.basename(image2_path)[:3]
                for d in range(20, 101):
                    dataset.append(Pair(fixed=fixed_subj, moving=moving_subj, depth=d))
        
        random.shuffle(dataset)
        #size = int(len(dataset) * train_split)
        if self.training:
            return dataset[:27680]
        return dataset[27680:30752]
