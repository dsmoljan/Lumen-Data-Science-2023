import torch
import torch.nn.functional as F
from src.models.abstract_model import AbstractModel
from torch import Tensor, nn


class CNN1DAudioNet(AbstractModel):
    def __init__(self, no_classes: 11):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=440, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=6)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(4)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(4)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(128)
        self.pool6 = nn.MaxPool1d(4)
        self.conv7 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn7 = nn.BatchNorm1d(256)
        self.pool7 = nn.MaxPool1d(4)
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
            self.conv7, self.relu, self.bn7, self.pool7
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