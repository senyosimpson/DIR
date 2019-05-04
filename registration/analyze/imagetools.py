import os
import skimage
import argparse
import numpy as np
import nibabel as nib
from nibabel.analyze import AnalyzeHeader

def _create_analyze_header(path):
    """
    Arguments:
        path (str) : path to analyze header file
    """
    with open(path, 'rb') as f:
        hdr = AnalyzeHeader(f.read())
        return hdr

def get_image_shape(hdr=None, path=None):
    """
    Arguments:
        hdr (AnalyzeHeader) : an analyze header object
    """
    assert hdr is None or path is None, 'One argument from {hdr, path} must be specified'
    if path:
        image_shape = _create_analyze_header(path).get_data_shape()
        return image_shape
    image_shape = hdr.get_data_shape()
    return image_shape

def get_image(path, image_shape):
    """
    Arguments:
        path (str) : path to analyze image
        image_shape (tuple) : shape of the output image
    """
    with open(path, 'rb') as f:
        image = np.fromfile(f, np.int16)
        image = image.reshape(image_shape)
        if len(image_shape) == 4:
            image = image.squeeze()
        return image

def resize_image(image, image_shape):
    """
    Arguments:
        image (np.ndarray) :
        image_shape (tuple) :
    """
    image = skimage.transform.resize(image, image_shape, anti_aliasing=True)
    return image.astype(np.float32)


def slice_image(image, dim, idx):
    """
    Arguments:
        image (np.ndarray) :
        dim (int) : the dimension to slice. Options are {0, 1, 2}
        idx (int) : the index with which to slice the image
    """
    assert dim in [0, 1 ,2], 'Dimension must be on of {0, 1, 2}'
    if dim == 0:
        return image[idx, :, :]
    elif dim == 1:
        return image[:, idx, :]
    elif dim == 2:
        return image[:, :, idx]
