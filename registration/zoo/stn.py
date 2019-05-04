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

        self.fc1 = nn.Linear(131072, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, self.nparam*2)
        self.tanh = nn.Tanh()

        self.fc3.weight.data.normal_(0, 1e-3)
        self.fc3.bias.data.zero_()

    def forward(self, deformation_field, moving):
        x = deformation_field.view(32, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.tanh(self.fc3(x))

        theta = x.view(-1, self.nparam, 2)
        grid = thinplate.tps_grid(theta, self.ctrl, (x.shape[0], ) + self.outshape)
        registered = F.grid_sample(moving, grid)
        return registered, theta
