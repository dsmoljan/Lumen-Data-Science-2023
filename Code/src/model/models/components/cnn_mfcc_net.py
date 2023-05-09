import torch.nn.functional as F
from src.model.models.abstract_model import AbstractModel
from torch import Tensor, nn


class CNN2DMfccNet(AbstractModel):
    def __init__(self, no_classes=11):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4))  # batch_size, 32, 37, 253
        self.bn1 = nn.BatchNorm2d(32)  # batch_size, 32, 37, 253
        self.pool1 = nn.MaxPool2d(2)  # batch_size, 32, 18, 126
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 10))  # batch_size, 64, 17, 117
        self.bn2 = nn.BatchNorm2d(64)  # batch_size, 64, 17, 117
        self.pool2 = nn.MaxPool2d(2)  # batch_size, 64, 8, 58
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 10))  # batch_size, 128, 7, 49
        self.bn3 = nn.BatchNorm2d(128)  # batch_size, 128, 7, 49
        self.pool3 = nn.MaxPool2d(2)  # batch_size, 128, 3, 24
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(2, 10))  # batch_size, 256, 2, 15
        self.bn4 = nn.BatchNorm2d(256)  # batch_size, 256, 2, 15
        self.pool4 = nn.MaxPool2d(2)  # batch_size, 256, 1, 7
        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, no_classes)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool1,
            self.conv2, self.bn2, self.relu, self.pool2,
            self.conv3, self.bn3, self.relu, self.pool3,
            self.conv4, self.bn4, self.relu, self.pool4
        )(x)
        x = F.avg_pool2d(x, x.size()[-2:])
        x = x.view(x.size(0), -1)
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
    _ = CNN2DMfccNet()