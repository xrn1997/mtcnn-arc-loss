import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),  # (10, 10, 10)
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # (10, 5, 5)
            nn.Conv2d(10, 16, 3, 1),  # (16, 3, 3)
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),  # (32, 1, 1)
            nn.BatchNorm2d(32),
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset