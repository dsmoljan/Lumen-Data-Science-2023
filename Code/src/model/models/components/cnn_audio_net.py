import torch
import torch.nn.functional as F
from src.model.models.abstract_model import AbstractModel
from torch import Tensor, nn


class CNN1DAudioNet(AbstractModel):
    def __init__(self, no_classes: 11):
        super().__init__()

        # this convolution expects inputs of duration 1s and sample rate 44100Hz
        # input shape: (batch_size, 1, 44100)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=440, stride=2)  # shape: (batch_size, 32, 21831)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)  # shape: (batch_size, 32, 5457)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=6)  # shape: (batch_size, 32, 5452)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)  # shape: (batch_size, 32, 1363)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)  # shape: (batch_size, 64, 1361)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(4)  # shape: (batch_size, 64, 340)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3)  # shape: (batch_size, 64, 338)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(4)  # shape: (batch_size, 64, 84)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3)  # shape: (batch_size, 128, 82)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(4)  # shape: (batch_size, 128, 20)
        self.conv6 = nn.Conv1d(128, 256, kernel_size=3)  # shape: (batch_size, 256, 18)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool6 = nn.MaxPool1d(4)  # shape: (batch_size, 256, 4)
        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, no_classes)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = nn.Sequential(
            self.conv1, self.relu, self.bn1, self.pool1,
            self.conv2, self.relu, self.bn2, self.pool2,
            self.conv3, self.relu, self.bn3, self.pool3,
            self.conv4, self.relu, self.bn4, self.pool4,
            self.conv5, self.relu, self.bn5, self.pool5,
            self.conv6, self.relu, self.bn6, self.pool6,
        )(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.squeeze(x, 2)
        x = nn.Sequential(self.fc1, self.dropout1, self.relu, self.fc2)(x)
        return x

    def get_cls_named_parameters(self):
        named_parameters = []
        for n, _ in self.fc1.named_parameters(prefix="fc1"):
            named_parameters.append(n)
        for n, _ in self.fc2.named_parameters(prefix="fc2"):
            named_parameters.append(n)
        return named_parameters


if __name__ == "__main__":
    _ = CNN1DAudioNet()