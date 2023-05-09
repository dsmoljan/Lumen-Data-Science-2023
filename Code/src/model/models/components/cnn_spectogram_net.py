from src.model.models.abstract_model import AbstractModel
from torch import Tensor, nn
from torchvision import models


class CNNSpectogramNet(AbstractModel):
    def __init__(self, no_classes = 11):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs_in = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs_in, no_classes))

    def forward(self, x: Tensor):
        return self.model(x)

    def get_cls_named_parameters(self):
        return [n for n, _ in self.model.fc.named_parameters(prefix="model.fc")]