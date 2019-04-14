""" STN using affine transformations.
Taken from (Spatial Transformer Network Tutorial)
[https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html] 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Localization(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.affine1 = nn.Linear((10 * 3 * 3), 32)
        self.affine2 = nn.Linear(32, 6)

        self.affine2.weight.data.zero_()
        self.affine2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, (10 * 3 * 3))
        x = self.affine1(x)
        x = self.affine2(x)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self):
        self.localization = self.Localization()

    def forward(self, x):
        affine_matrix = self.localization(x)
        affine_matrix = affine_matrix.view(-1, 2, 3)
        grid = F.affine_grid(affine_matrix, x.size())
        transformed_x = F.grid_sample(x, grid)
        return transformed_x