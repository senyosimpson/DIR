""" Implementation of the network from the paper [Unsupervised End-to-end Learning
for Deformable Medical Image Registration](https://arxiv.org/abs/1711.08608)
"""
import yaml
import torch
import torch.nn as nn
from registration.utils.model import build_model


class WarpNet(nn.Module):
    def __init__(self, architecture, stn):
        """
        Args:
            architecture (str) : path to file that describes
                architecture of the neural network written
                in yaml format
        """
        super().__init__()
        self.architecture = architecture
        self.model = build_model(self.architecture)
        self.stn = stn

    def forward(self, fixed, moving):
        x = torch.cat((fixed, moving), dim=1)
        x = self.model(x)
        x = self.stn(x)
        return x
