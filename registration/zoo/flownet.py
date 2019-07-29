'''
Portions of this code copyright 2017, Clement Pinard
Model taken from NVIDIA PyTorch implementation of FlowNet2S
[FlowNet2 PyTorch](https://github.com/NVIDIA/flownet2-pytorch)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import numpy as np

from .submodules import *

class FlowNetS(nn.Module):
    def __init__(self, stn, input_channels=2, use_batchnorm=True):
        super(FlowNetS,self).__init__()

        self.stn = stn
        self.use_batchnorm = use_batchnorm
        self.conv0   = conv(self.use_batchnorm, input_channels, 32, kernel_size=7) 
        self.conv1   = conv(self.use_batchnorm,  32,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.use_batchnorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.use_batchnorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.use_batchnorm, 256,  256)
        self.conv4   = conv(self.use_batchnorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.use_batchnorm, 512,  512)
        self.conv5   = conv(self.use_batchnorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.use_batchnorm, 512,  512)
        self.conv6   = conv(self.use_batchnorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.use_batchnorm,1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(194, 32)
        self.deconv0  = deconv(98, 2)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(98)
        self.predict_flow0 = predict_flow(36)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, image_pair):
        out_conv0 = self.conv0(image_pair)
        out_conv1 = self.conv1(out_conv0)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up),1)
        flow2       = self.predict_flow2(concat2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)
        
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1       = self.predict_flow1(concat1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        flow0       = self.predict_flow0(concat0,)

        deformation_field = flow0
        
        moving = image_pair[:,1:2:,:]
        registered, theta = self.stn(deformation_field, moving)
        return registered, theta, deformation_field

        # For use later when doing multi scale loss
        # if self.training:
            #return (flow1_up, flow1, flow2,flow3,flow4,flow5,flow6)
        #else:
            #return flow1_up

