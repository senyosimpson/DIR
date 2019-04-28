""" Implementation of the network from the paper [Unsupervised End-to-end Learning
for Deformable Medical Image Registration](https://arxiv.org/abs/1711.08608)
"""
import yaml
import torch
import torch.nn as nn

from .submodules import conv, deconv


class WarpNet(nn.Module):
    def __init__(self, stn, use_batchnorm=True):
        super().__init__()
        self.stn = stn

        self.conv1 = conv(use_batchnorm, 2, 64, stride=2)
        self.conv2 = conv(use_batchnorm, 64, 128, stride=2)
        self.conv3 = conv(use_batchnorm, 128, 256, stride=2)
        self.conv3_1 = conv(use_batchnorm, 256, 256)
        self.conv4 = conv(use_batchnorm, 256, 512, stride=2)
        self.conv4_1 = conv(use_batchnorm, 512, 512)
        self.conv5 = conv(use_batchnorm, 512, 512, stride=2)
        self.conv5_1 = conv(use_batchnorm, 512, 512)
        self.conv6 = conv(use_batchnorm, 512, 1024, stride=2)
        self.conv6_1 = conv(use_batchnorm, 1024, 1024)

        self.deconv6 = deconv(1024, 512)
        self.deconv5 = deconv(512, 256)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 2)
        self.deconv1 = deconv(2, 1)

        # define upsample layers to get losses

    def forward(self, fixed, moving):
        x = torch.stack((fixed, moving), dim=1)
        moving = moving.reshape(32, 1, 256, 256)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        x = self.conv6_1(x)

        x = self.deconv6(x)
        x = self.deconv5(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        transformed_x = self.stn(x, moving)
        return transformed_x
