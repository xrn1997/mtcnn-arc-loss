import torch
import torch.nn as nn

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),  # (28, 22, 22)
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, 1),  # (28, 11, 11)
            nn.Conv2d(28, 48, 3, 1),  # (48, 9, 9)
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),  # (48, 4, 4)
            nn.Conv2d(48, 64, 2, 1),  # (64, 3, 3)
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64 * 3 * 3, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)

        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset
