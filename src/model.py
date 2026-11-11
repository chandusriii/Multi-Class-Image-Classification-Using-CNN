"""CNN architecture used for CIFAR-10 classification."""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, images):
        features = self.conv_layers(images)
        features = features.view(features.size(0), -1)
        return self.fc_layers(features)
