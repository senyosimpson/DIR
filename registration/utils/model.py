""" Utility functions for models """
import yaml
import torch
import torch.nn as nn

from registration.units import ResidualUnit

LAYER_TYPES = {
    'conv' : nn.Conv2d,
    'convtranspose': nn.ConvTranspose2d,
    'affine' : nn.Linear,
    'residual' : ResidualUnit
    }

ACTIVATION_TYPES = {
    'relu' : nn.ReLU
    }

def build_model(architecture):
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
            layer = LAYER_TYPES[layer_def['type']]
            activation = ACTIVATION_TYPES[layer_def['activation']]

            if layer_def['type'] == 'residual' :
                conv1_params = layer_def['conv1_params']
                conv2_params = layer_def['conv2_params']
                conv_params = (conv1_params, conv2_params)
                bn1_features = layer_def['bn1_features']
                bn2_features = layer_def['bn2_features']
                bn_features = (bn1_features, bn2_features)
                shortcut_params = layer_def['shortcut']
                shortcut = nn.Sequential(
                    nn.Conv2d(**shortcut_params)
                )
                layers.append(layer(conv_params, bn_features, activation, shortcut))
            else:
                params = layer_def['params']
                bn_features = layer_def['bn_features']
                sequential = nn.Sequential(
                    layer(**params),
                    nn.BatchNorm2d(bn_features),
                    activation(inplace=True)
                )
                layers.append(sequential)

        return nn.Sequential(*layers)