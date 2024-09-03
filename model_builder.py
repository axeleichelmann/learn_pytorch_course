"""
This file contains the TinyVGG model class
"""

import torch
import torch.nn as nn


class TinyVGG(nn.Module):
  """
  Creates the TinyVGG CNN architecture

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  ---------------------------------------------------------
  Args:

  input_shape - the number of input channels
  output_shape - the number of output channels
  hidden_channels - the number of channels in the hidden layers

  """

  def __init__(self, input_shape : int, output_shape : int, hidden_channels : int=10):
    super().__init__()

    self.conv_layer1 = nn.Sequential(
                                    nn.Conv2d(in_channels=input_shape, out_channels=hidden_channels,
                                              kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                              kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
                                    )

    self.conv_layer2 = nn.Sequential(
                                    nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                              kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                              kernel_size=3, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
                                    )

    self.classifier_layer = nn.Sequential(
                                          nn.Flatten(),
                                          nn.Linear(in_features=hidden_channels*13*13, out_features=output_shape)
                                          )

  def forward(self, x):
    return self.classifier_layer(self.conv_layer2(self.conv_layer1(x)))
