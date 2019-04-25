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


class ResidualUnit(nn.Module):
    def __init__(self, params, bn_features, activation, shortcut=None):
        super().__init__()
        self.conv1_params, self.conv2_params = params
        self.bn1_features, self.bn2_features = bn_features
        self.conv1 = nn.Conv2d(**self.conv1_params)
        self.conv2 = nn.Conv2d(**self.conv2_params)
        self.bn1 = nn.BatchNorm2d(self.bn1_features)
        self.bn2 = nn.BatchNorm2d(self.bn2_features)
        self.activation = activation()
        self.shortcut = shortcut

    def forward(self, x):
        input_tensor = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.shortcut:
            input_tensor = self.shortcut(input_tensor)
        x = torch.add(x, input_tensor)
        x = self.activation(x)
        return x