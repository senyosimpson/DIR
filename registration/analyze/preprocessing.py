""" Preprocessing for LPBA40 dataset """
import os
import h5py
import random
import logging
import argparse
import numpy as np
from glob import glob
from registration.analyze import imagetools


if __name__ == "__main__":
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        type=str,
                        required=True,
                        help='root directory to Analyze images')
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='name of the output file for the dataset')
    parser.add_argument('--header',
                        type=str,
                        required=True,
                        help='path to header file')
    
    args = parser.parse_args()
    root = args.root
    out = args.out
    hdr = args.header

    image_paths = glob(os.path.join(root, '*.img'))
    # get image shape, assumption is all images have the same shape
    image_shape = imagetools.get_image_shape(path=hdr)

    logger.info('MAIN SCRIPT STARTED')
    logger.info('Creating h5 dataset')
    with h5py.File(out, 'w') as f:
        f.create_dataset(
            name='Brain MRI Dataset',
            shape=(148200, 2, 256, 256),
            dtype=np.float64
        )
        pos = 0
        for idx, image1_path in enumerate(image_paths):
            logger.info('Creating pairs for image %d of %d' % (idx+1, len(image_paths)))
            image1 = imagetools.get_image(image1_path, image_shape)
            image1 = np.pad(image1,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
            image1 = imagetools.resize_image(image1, (256,256,256))
            for image2_path in image_paths[idx:]:
                image2 = imagetools.get_image(image2_path, image_shape)
                image2 = np.pad(image1,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
                image2 = imagetools.resize_image(image2, (256,256,256))
                for dim in (0, 1, 2):
                    for i in range(60, 190): # chosen so that all dimensions have useful images
                        image_pair = np.array([
                                        imagetools.slice_image(image1, dim=dim, idx=i),
                                        imagetools.slice_image(image2, dim=dim, idx=i)
                                    ])
                        assert image_pair.shape == (2, 256, 256), \
                            'Image pair shape is incorrect. Should be (2, 256, 256)'
                        f['Brain MRI Dataset'][i, :, :, :] = image_pair
                        pos += 1
        