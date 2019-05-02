""" Preprocessing for LPBA40 dataset """
import os
import h5py
import random
import logging
import argparse
import numpy as np
from glob import glob
from registration.analyze import imagetools

DIM_RANGE = {
    0: (4, 211), 
    1: (40, 231), 
    2: (70, 191)}

if __name__ == "__main__":
    logger = logging.getLogger('preprocessing')
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
            shape=(72580, 2, 256, 256),
            dtype=np.float32,
            chunks=(32,2,256,256)
        )
        pos = 0
        for idx, image1_path in enumerate(image_paths):
            logger.info('Creating pairs for image %d of %d' % (idx+1, len(image_paths)))
            image1 = imagetools.get_image(image1_path, image_shape)
            image1 = np.pad(image1,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
            image1 = imagetools.resize_image(image1, (256,256,256))
            pair_image_paths = filter(lambda x: x != image1_path, image_paths)
            for image2_path in pair_image_paths:
                image2 = imagetools.get_image(image2_path, image_shape)
                image2 = np.pad(image2,((0,0), (2,2), (0,0)), mode='constant', constant_values=0)
                image2 = imagetools.resize_image(image2, (256,256,256))
                # using only 1 dim to simplify the learning problem initially 
                dim = 1
                start, end = DIM_RANGE[dim]
                for i in range(start, end):
                    image_pair = np.array([
                                    imagetools.slice_image(image1, dim=dim, idx=i),
                                    imagetools.slice_image(image2, dim=dim, idx=i)
                                ])
                    assert image_pair.shape == (2, 256, 256), \
                        'Image pair shape is incorrect. Should be (2, 256, 256)'
                    f['Brain MRI Dataset'][pos, ...] = image_pair
                    pos += 1


        logger.info('Shuffling Dataset')
        random.shuffle(f['Brain MRI Dataset'])
    logger.info('Dataset Created')