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
from registration.zoo import FlowNetS
from registration.zoo import STN
from registration.losses import SmoothingLoss
from datetime import datetime

if __name__ == '__main__':
    date = datetime.today().strftime('%m_%d')

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
    parser.add_argument('--pretrained',
                        type=str,
                        required=False,
                        help='path to use pretrained network')
    parser.add_argument('--logdir',
                        type=str,
                        required=True,
                        help='path to save model checkpoints')
    parser.add_argument('--seed',
                        type=int,
                        required=False,
                        help='value to set random seed')
    parser.add_argument('--grid_size',
                        type=int,
                        required=False,
                        default=6,
                        help='(grid size x grid size) control points')
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
    parser.add_argument('--num_workers',
                        type=int,
                        required=False,
                        default=4,
                        help='number of workers for data loading')
    args = parser.parse_args()

    logger.info('MAIN SCRIPT STARTED')
    use_cuda = not False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    grid_size = (args.grid_size, args.grid_size)
    spatial_transformer = STN(ctrlshape=grid_size).to(device)
    model = FlowNetS(stn=spatial_transformer).to(device)

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained), strict=False)
    elif args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    model.train()
    optimizer = optim.Adam(list(model.parameters()) + list(spatial_transformer.parameters()),
                           lr=1e-5, betas=(0.9, 0.999),
                           weight_decay=0.0005)

    photometric_diff_loss = nn.L1Loss()
    smoothing_loss = SmoothingLoss()
    # define smoothing loss

    tsfm = transforms.Compose([
                        Normalize(),
                        ToTensor()
                        ])
    
    mri = LPBA40(args.dataset, seed=args.seed, transform=tsfm)
    train_loader = torch.utils.data.DataLoader(
                                mri,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # details of training
    logger.info('')
    logger.info('Size of Training Dataset : %d' % (len(train_loader) * args.batch_size))
    logger.info('Batch Size : %d' % args.batch_size)
    logger.info('Number of Epochs : %d' % args.epochs)
    logger.info('Steps per Epoch : %d' % len(train_loader))
    logger.info('')

    start_epoch = start_epoch if args.load_checkpoint else 0
    for epoch in range(start_epoch, args.epochs):
        mean_loss = 0
        logger.info('============== Epoch %d/%d ==============' % (epoch+1, args.epochs))
        for batch_idx, image_pair in enumerate(train_loader):
            image_pair = image_pair.to(device)
            optimizer.zero_grad()
            registered, _, deformation_field = model(image_pair)

            # calculate photometric diff loss and smoothing loss
            alpha = 1 # weight term for the photometric diff loss
            beta = 0.05 # weight term for the smoothing loss
            fixed = image_pair[:,0:1,:,:]
            pd_loss = photometric_diff_loss(registered, fixed)
            s_loss = smoothing_loss(deformation_field)
            loss = alpha * pd_loss # + (beta * s_loss)
            loss.backward()
            optimizer.step()

            logger.info('step: %d, loss: %.3f' % (batch_idx, loss.item()))
            mean_loss += loss.item()
        
        logger.info('epoch : %d, average loss : %.3f' % (epoch+1, mean_loss/len(train_loader)))

        save_path =  'warpnet_mri_checkpoint_%d_%s%s' % (epoch+1, date, '.pt')
        if args.pretrained:
            save_path =  'warpnet_mri_pretrained_checkpoint_%d_%s%s' % (epoch+1, date, '.pt')

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                f = os.path.join(args.logdir, save_path)) 
        logger.info('Checkpoint saved to %s' % save_path)

    logger.info('Training Complete')