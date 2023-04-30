import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CNN1DAudioFeaturesNet(nn.Module):
    def __init__(self, no_classes):
        super().__init__()

        # input shape: (batch_size, 46, 259)
        self.conv1 = nn.Conv1d(46, 32, kernel_size=8)  # (batch_size, 32, 252)
        self.bn1 = nn.BatchNorm1d(32)  # (batch_size, 32, 252)
        self.pool1 = nn.MaxPool1d(2)  # (batch_size, 32, 126)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=4)  # (batch_size, 64, 123)
        self.bn2 = nn.BatchNorm1d(64)  # (batch_size, 64, 123)
        self.pool2 = nn.MaxPool1d(2)  # (batch_size, 64, 61)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=2)  # (batch_size, 128, 60)
        self.bn3 = nn.BatchNorm1d(128)  # (batch_size, 128, 60)
        self.pool3 = nn.MaxPool1d(2)  # (batch_size, 128, 30)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=2)  # (batch_size, 256, 29)
        self.bn4 = nn.BatchNorm1d(256)  # (batch_size, 256, 29)
        self.pool4 = nn.MaxPool1d(2)  # (batch_size, 256, 14)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=2)  # (batch_size, 512, 13)
        self.bn5 = nn.BatchNorm1d(512)  # (batch_size, 512, 13)
        self.pool5 = nn.MaxPool1d(2)  # (batch_size, 512, 6)

        self.fc1 = nn.Linear(512, 64)  # (batch_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, no_classes)  # (batch_size, no_classes)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = nn.Sequential(
            self.conv1, self.relu, self.bn1, self.pool1,
            self.conv2, self.relu, self.bn2, self.pool2,
            self.conv3, self.relu, self.bn3, self.pool3,
            self.conv4, self.relu, self.bn4, self.pool4,
            self.conv5, self.relu, self.bn5, self.pool5
            )(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.squeeze(x, -1)
        x = nn.Sequential(self.fc1, self.dropout1, self.relu, self.fc2)(x)
        return x
