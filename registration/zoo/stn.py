import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from registration.warp import thinplate


class SpatialTransformer(nn.Module):
    def __init__(self, outshape=(1, 256, 256), ctrlshape=(6, 6)):
        super().__init__()
        self.nctrl = np.prod(ctrlshape)
        self.outshape = outshape
        self.nparam = self.nctrl + 2
        ctrl = thinplate.uniform_grid(ctrlshape)
        self.register_buffer('ctrl', ctrl.view(-1,2))

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, self.nparam*2)
        self.tanh = nn.Tanh()

        self.fc2.weight.data.normal_(0, 1e-3)
        self.fc2.bias.data.zero_()

    def forward(self, deformation_field, moving):
        x = F.relu(F.max_pool2d(self.conv1(deformation_field), kernel_size=6, padding=3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=6, padding=3))
        x = x.view(32, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.tanh(self.fc2(x))

        theta = x.view(-1, self.nparam, 2)
        grid = thinplate.tps_grid(theta, self.ctrl, (x.shape[0], ) + self.outshape)
        registered = F.grid_sample(moving, grid)
        return registered