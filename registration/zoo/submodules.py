import torch
import torch.nn as nn

def conv(batchNorm, in_channels, out_channels, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True)
        )

def deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True)
    )

def predict_flow(in_planes, out_planes=2):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=True)
