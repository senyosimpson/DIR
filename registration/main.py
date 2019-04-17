import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from registration.data.datasets import MNIST
from registration.data.transforms import ToTensor, Normalize
from registration.zoo import WarpNet
from registration.zoo import AffineTransformer, ThinPlateTransformer


if __name__ == '__main__':
    logger = logging.getLogger('dir')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='path to dataset')
    parser.add_argument('--ds_size',
                        type=int,
                        required=False,
                        default=60000,
                        help='size of dataset to generate')
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
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=10,
                        help='training batch size')
    args = parser.parse_args()

    logger.info('')
    logger.info('MAIN SCRIPT STARTED')

    spatial_transformer = ThinPlateTransformer(
                            args.stn_type,
                            args.image_height,
                            args.image_width,
                            args.grid_size,
                            args.span_range)

    warpnet = WarpNet(args.arch, spatial_transformer)
    logger.info('Model Created')

    use_cuda = not False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.SGD(warpnet.parameters(), lr=0.01, momentum=0.9)
    photometric_diff_loss = nn.L1Loss()
    # smoothing_loss = None
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tsfm = transforms.Compose([
                        Normalize(),
                        ToTensor()
                        ])
    
    mnist = MNIST(args.dataset, args.ds_size, transform=tsfm)
    train_loader = torch.utils.data.DataLoader(
                                mnist,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)

for batch_idx, (fixed, moving) in enumerate(train_loader):
    fixed, moving = fixed.to(device), moving.to(device)
    optimizer.zero_grad()
    output = warpnet(fixed, moving)

    # calculate photometric diff loss and smoothing loss
    alpha = 1 # weight term for the photometric diff loss
    beta = 0.5 # weight term for the smoothing loss
    pdl = photometric_diff_loss(output, fixed)
    #sl = smoothing_loss(output, target)
    total_loss = alpha*pdl #+ beta*sl
    total_loss.backward()
    optimizer.step()
    logger.info('loss: %.3f' % total_loss.item())
