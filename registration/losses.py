import torch
import torch.nn as nn
import torch.nn.functional as F
from registration.warp import thinplate

class MultiScaleLoss(nn.Module):
    def __init__(self, ctrlshape=(6, 6)):
        self.pd_loss = nn.L1Loss()
        self.outshapes = [(2**x, 2**x) for x in range(2,9)]
        ctrl = thinplate.uniform_grid(ctrlshape)
        self.register_buffer('ctrl', ctrl.view(-1,2))
    
    def forward(self, fixed, moving, thetas):
        loss = 0
        for outshape, theta in zip(self.outshapes, thetas):
            grid = thinplate.tps_grid(theta, self.ctrl, outshape)
            registered = F.grid_sample(moving, grid)
            scale_factor = outshape[0]/256
            fixed = F.interpolate(fixed, scale_factor=scale_factor)
            loss += self.pd_loss(registered, fixed)

        return loss