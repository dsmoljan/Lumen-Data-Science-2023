from torch import nn, Tensor
from torchvision import models

class CNNSpectogramNet(nn.Module):
    def __init__(self, no_classes = 11):
        super().__init__()
        # TODO: baza modela se isto može učitati iz Hydre
        self.model = models.resnet50(pretrained=True)
        num_ftrs_in = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs_in, no_classes))

    def forward(self, x: Tensor):
        return self.model(x)



