import torch
import torch.nn as nn


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),  # (32, 46, 46)
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, 1),  # (32, 23, 23)
            nn.Conv2d(32, 64, 3, 1),  # (64, 21, 21)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),  # (64, 10, 10)
            nn.Conv2d(64, 64, 3, 1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # (64, 4, 4)
            nn.Conv2d(64, 128, 2, 1),  # (128, 3, 3)
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)  # conv5
        self.prelu5 = nn.PReLU()  # prelu5
        # detection
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        label = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return label, offset