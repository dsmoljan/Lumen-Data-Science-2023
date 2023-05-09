import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.model.models.abstract_model import AbstractModel



class CNN1DAudioFeaturesNet(AbstractModel):
    def __init__(self, no_classes):
        super().__init__()

        # input shape: (batch_size, 40, 87)
        self.conv1 = nn.Conv1d(40, 32, kernel_size=8)  # (batch_size, 32, 80)
        self.bn1 = nn.BatchNorm1d(32)  # (batch_size, 32, 80)
        self.pool1 = nn.MaxPool1d(2)  # (batch_size, 32, 40)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)  # (batch_size, 64, 39)
        self.bn2 = nn.BatchNorm1d(64)  # (batch_size, 64, 39)
        self.pool2 = nn.MaxPool1d(2)  # (batch_size, 64, 19)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=2)  # (batch_size, 128, 18)
        self.bn3 = nn.BatchNorm1d(128)  # (batch_size, 128, 18)
        self.pool3 = nn.MaxPool1d(2)  # (batch_size, 128, 9)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=2)  # (batch_size, 256, 8)
        self.bn4 = nn.BatchNorm1d(256)  # (batch_size, 256, 8)
        self.pool4 = nn.MaxPool1d(2)  # (batch_size, 256, 4)

        self.fc1 = nn.Linear(256, 64)  # (batch_size, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, no_classes)  # (batch_size, no_classes)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = nn.Sequential(
            self.conv1, self.relu, self.bn1, self.pool1,
            self.conv2, self.relu, self.bn2, self.pool2,
            self.conv3, self.relu, self.bn3, self.pool3,
            self.conv4, self.relu, self.bn4, self.pool4,
            )(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.squeeze(x, -1)
        x = nn.Sequential(self.fc1, self.dropout1, self.relu, self.fc2)(x)
        return x

    def get_cls_named_parameters(self):
        named_parameters = []
        for n, _ in self.fc1.named_parameters(prefix="fc1"):
            named_parameters.append(n)
        for n, _ in self.fc2.named_parameters(prefix="fc2"):
            named_parameters.append(n)
        return named_parameters
