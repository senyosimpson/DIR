import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from registration.zoo import WarpNet
from registration.zoo import SpatialTransformer


if __name__ == '__main__':
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        type=str,
                        required=True,
                        help='path to architecture ')
    parser.add_argument('--stn_type',
                        type=str,
                        required=False,
                        default='unbounded',
                        help='unbounded or bounded stn type')
    parser.add_argument('--image_height',
                        type=int,
                        required=True,
                        help='height of image')
    parser.add_argument('--image_width',
                        type=int,
                        required=True,
                        help='width of image')
    parser.add_argument('--grid_size',
                        type=int,
                        required=False,
                        default=4,
                        help='(grid size x grid size) control points')
    parser.add_argument('--span_range',
                        type=int,
                        required=False,
                        default=0.9,
                        help='percentage of image dimensions to span over')
    args = parser.parse_args()

    logging.info('')
    logging.info('MAIN SCRIPT STARTED')

    spatial_transformer = SpatialTransformer(
                            args.stn_type,
                            args.image_height,
                            args.image_width,
                            args.grid_size,
                            args.span_range)

    warpnet = WarpNet(args.arch, spatial_transformer)
    logging.info('Model Created')

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.SGD(warpnet.parameters(), lr=0.01, momentum=0.9)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tsfm = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])
    train_loader = torch.utils.data.DataLoader(
                            MNIST('data',
                                  train=True,
                                  download=False,
                                  transform=tsfm,
                                  batch_size=32,
                                  shuffle=True,
                                  **kwargs))

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = warpnet(data)
    # define loss functions