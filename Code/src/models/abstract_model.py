from torch import nn


class AbstractModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_cls_named_parameters(self):
        """
        Returns named parameters of the classifier of the model.
        """
        raise NotImplementedError
