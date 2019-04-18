import torch
import torch.nn as nn


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