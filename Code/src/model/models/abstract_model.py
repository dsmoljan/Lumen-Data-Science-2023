from torch import nn


class AbstractModel(nn.Module):
    """
    An abstract class which defines two base methods all subclasses must implement. If not implemented and called,
    an error will be raised.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def get_cls_named_parameters(self):
        """
        Returns named parameters of the classifier of the model. This method should be implemented if the subclass
        is expected to work with multiple optimizers.
        """
        raise NotImplementedError