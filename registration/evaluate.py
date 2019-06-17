import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from registration.data.datasets import LPBA40
from registration.data.transforms import ToTensor, Normalize, Transpose
from registration.zoo import FlowNetS
from registration.zoo import STN
from registration.warp import thinplate
from registration.metrics import mutual_information, jaccard_coeff
from datetime import datetime

class Warper(nn.Module):
    def __init__(self, outshape=(32, 1, 256, 256), ctrlshape=(6, 6)):
        super().__init__()
        self.nctrl = np.prod(ctrlshape)
        self.outshape = outshape
        self.nparam = self.nctrl + 2
        ctrl = thinplate.uniform_grid(ctrlshape)
        self.register_buffer('ctrl', ctrl.view(-1,2))

    def forward(self, theta, mask):
        grid = thinplate.tps_grid(theta, self.ctrl, self.outshape)
        registered = F.grid_sample(mask, grid)
        return registered

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
    parser.add_argument('--load_model',
                        type=str,
                        required=True,
                        help='path to model weights to load')
    parser.add_argument('--out',
                        type=str,
                        required=False,
                        default='results.csv',
                        help='output csv file name for saving results')
    parser.add_argument('--seed',
                        type=int,
                        required=False,
                        help='value to set random seed')
    parser.add_argument('--grid_size',
                        type=int,
                        required=False,
                        default=6,
                        help='(grid size x grid size) control points')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=32,
                        help='test batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        required=False,
                        default=4,
                        help='number of workers for data loading')
    args = parser.parse_args()

    logger.info('MAIN SCRIPT STARTED')
    use_cuda = not False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    warper = Warper().to(device)

    grid_size = (args.grid_size, args.grid_size)
    spatial_transformer = STN(ctrlshape=grid_size)
    spatial_transformer = spatial_transformer.to(device)
    model = FlowNetS(stn=spatial_transformer)

    weights = torch.load(args.load_model)
    model.load_state_dict(weights['model_state_dict'])
    model.to(device)
    model.eval()


    tsfm = transforms.Compose([
                        Normalize(training=False),
                        ToTensor(training=False)
                        ])
    
    mri = LPBA40(args.dataset, seed=args.seed, training=False, transform=tsfm)
    test_loader = torch.utils.data.DataLoader(
                                mri,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # details of evaluation
    logger.info('')
    logger.info('Size of Evaluation Dataset : %d' % (len(test_loader) * args.batch_size))
    logger.info('Batch Size : %d' % args.batch_size)
    logger.info('Number of steps : %d' % len(test_loader))
    logger.info('')


    mi = []
    jacc = []
    logger.info('============== Starting Evaluation ==============')
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            logger.info('Batch %d/%d' % (batch_idx+1, len(test_loader)))
            image_pair = images[:,:2,:,:]
            fixed_mask, moving_mask = images[:,2,:,:], images[:,3:4,:,:] 
            image_pair = image_pair.to(device)
            registered, theta, _ = model(image_pair) 
            moving_mask = moving_mask.to(device)
            registered_mask = warper(theta, moving_mask) 

            # evaluate
            fixed_mask = fixed_mask.cpu().numpy()
            registered_mask = registered_mask.squeeze().cpu().numpy() 
            registered_mask = np.around(registered_mask)
            jacc += [jaccard_coeff(fixed_mask[idx], registered_mask[idx]) for idx in range(args.batch_size)]

            fixed = image_pair[:,0,:,:].cpu().numpy() 
            registered = registered.squeeze().cpu().numpy() 
            mi += [mutual_information(fixed[idx], registered[idx]) for idx in range(args.batch_size)]

    mean_mi = np.array(mi).mean()
    mean_jacc = np.array(jacc).mean()

    metric_names = ['Mutual Information', 'Jaccard Coefficient']
    metric_scores = [mean_mi, mean_jacc]

    df = pd.DataFrame({metric : [score] for metric, score in zip(metric_names, metric_scores)})
    df.to_csv(args.out)
    logger.info('Mean Mutual Information : %s' % mean_mi)
    logger.info('Mean Jaccard Coefficient : %s' % mean_jacc)
    logger.info("Evaluation Complete")