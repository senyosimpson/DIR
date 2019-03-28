""" Implementation of the network from the paper [Unsupervised End-to-end Learning
for Deformable Medical Image Registration](https://arxiv.org/abs/1711.08608)
"""
import yaml
import torch
import torch.nn as nn

class ResidualUnit(nn.Module):
    def __init__(self, params, bn_features, activation, shortcut=None):
        super().__init__()
        self.params1, self.params2 = params
        self.bn_features1, self.bn_features2 = bn_features
        self.conv1 = nn.Conv2d(**self.params1)
        self.conv2 = nn.Conv2d(**self.params2)
        self.bn1 = nn.BatchNorm2d(self.bn_features1)
        self.bn2 = nn.BatchNorm2d(self.bn_features2)
        self.activation = activation()
        self.shortcut = shortcut

    def forward(self, x):
        input_tensor = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.conv2(x)
        if self.shortcut:
            input_tensor = self.shortcut(input_tensor)
        x = torch.add(x, input_tensor)
        x = self.activation(x)
        return x


class WarpNet(nn.Module):
    def __init__(self, architecture):
        """
        Args:
            architecture (str) : path to file that describes
                architecture of the neural network written
                in yaml format
        """
        super().__init__()
        self.layer_types = {
            'conv' : nn.Conv2d,
            'deconv': nn.ConvTranspose2d,
            'affine' : nn.Linear,
            'residual' : ResidualUnit,
        }
        self.activations = {
            'relu' : nn.ReLU
        }
        self.architecture = architecture
        self.model = self._build_model(self.architecture)


    def _build_model(self, architecture):
        """ Build the model from yaml specification
        Args:
            architecture (str) : path to file that describes
            the architecture of the neural network written
            in yaml format
        """
        with open(architecture) as f:
            archi = yaml.load(f)

        layers = []
        for layer_def in archi.values():
            print(layer_def)
            layer = self.layer_types[layer_def['type']]
            activation = self.activations[layer_def['activation']]

            if layer_def['type'] == 'residual' :
                params1 = layer_def['params1']
                params2 = layer_def['params2']
                params = (params1, params2)
                bn_features1 = layer_def['bn_features1']
                bn_features2 = layer_def['bn_features2']
                bn_features = (bn_features1, bn_features2)
                shortcut_params = layer_def['shortcut']
                shortcut = nn.Sequential(
                    nn.Conv2d(**shortcut_params)
                )
                layers.append(layer(params, bn_features, activation, shortcut))
            else:
                params = layer_def['params']
                bn_features = layer_def['bn_features']
                sequential = nn.Sequential(
                    layer(**params),
                    nn.BatchNorm2d(bn_features),
                    activation()
                )
                layer.append(sequential)

        return nn.Sequential(*layers)


    def forward(self, fixed, moving):
        model_input = torch.cat([fixed, moving], dim=1)
        x = self.model(model_input)
        # add stn module here
        return x
