import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from registration.data.datasets import LPBA40
from registration.data.transforms import ToTensor, Normalize, Transpose
from registration.zoo import WarpNet
from registration.zoo import AffineTransformer, ThinPlateTransformer
from datetime import datetime

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
    parser.add_argument('--load_checkpoint',
                        type=str,
                        required=False,
                        help='path to checkpoint to load')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='path to save model checkpoints')
    parser.add_argument('--stn_type',
                        type=str,
                        required=False,
                        default='bounded',
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
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=10,
                        help='number of epochs when training')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=32,
                        help='training batch size')
    args = parser.parse_args()

    logger.info('MAIN SCRIPT STARTED')
    use_cuda = not False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    spatial_transformer = ThinPlateTransformer(
                            args.stn_type,
                            args.image_height,
                            args.image_width,
                            args.grid_size,
                            args.span_range)


    model = WarpNet(stn=spatial_transformer)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0005)
    photometric_diff_loss = nn.L1Loss()
    # define smoothing loss

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    tsfm = transforms.Compose([
                        Normalize(),
                        ToTensor()
                        ])
    
    mri = LPBA40(args.dataset, transform=tsfm)
    train_loader = torch.utils.data.DataLoader(
                                mri,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1)

# details of training
logger.info('')
logger.info('Size of Training Dataset : %d' % (len(train_loader) * args.batch_size))
logger.info('Batch Size : %d' % args.batch_size)
logger.info('Number of Epochs : %d' % args.epochs)
logger.info('Steps per Epoch : %d' % len(train_loader))
logger.info('')

date = datetime.today().strftime('%m_%d')

for epoch in range(args.epochs):
    logger.info('============== Epoch %d/%d ==============' % (epoch+1, args.epochs))
    for batch_idx, (fixed, moving) in enumerate(train_loader):
        fixed, moving = fixed.to(device), moving.to(device)
        optimizer.zero_grad()
        output = model(fixed, moving)

        # calculate photometric diff loss and smoothing loss
        alpha = 10 # weight term for the photometric diff loss
        beta = 0.5 # weight term for the smoothing loss
        pd_loss = photometric_diff_loss(output, fixed)
        #s_loss = smoothing_loss(output, target)
        loss = alpha * pd_loss #+ beta*sl
        loss.backward()
        optimizer.step()
        logger.info('step: %d, loss: %.3f' % (batch_idx, loss.item()))

    save_path =  'warpnet_mri_checkpoint_%d_%s%s' % (epoch+1, date, '.pt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            f = os.path.join(args.logdir, save_path)) 
    logger.info('Checkpoint saved to %s' % save_path)