from src.model.models.abstract_model import AbstractModel
from torch import Tensor, nn
from torchvision import models


class CNNSpectogramNet(AbstractModel):
    """
    ResNet50 model on spectograms.

    Args:
        no_classes (int): Number of classes in the dataset.
    """
    def __init__(self, no_classes = 11):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs_in = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs_in, no_classes))

    def forward(self, x: Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor (logits).
        """
        return self.model(x)

    def get_cls_named_parameters(self):
        """
        Returns:
            list: List of tuples of the form (name, parameter) for the classifier.
        """
        return [n for n, _ in self.model.fc.named_parameters(prefix="model.fc")]