import torch
from src.model.models.abstract_model import AbstractModel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


class AST(AbstractModel):
    """
    AST model class. This class is a wrapper around the transformers library.

    Args:
        no_classes (int): Number of classes in the dataset. \\
        mean (float): Spectrogram mean of the dataset. \\
        std (float): Spectrogram standard deviation of the dataset. \\
        max_length (int): Maximum length of the specrograms in the dataset. 1024 is a default value. If a different value is give, pre-trained positional embeddings will be discarded and new will be initialized. \\
        model_name_or_path (str): Name or path of the model to be used. If a name is given, the model will be downloaded from the HuggingFace repository. If a path is given, the model will be loaded from the path.
    """
    def __init__(self, no_classes, mean, std, max_length, model_name_or_path):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=no_classes, return_dict=True,
                                                 max_length=max_length)
        self.featurizer = AutoFeatureExtractor.from_pretrained(model_name_or_path, mean=mean, std=std,
                                                               max_length=max_length)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, config=self.config,
                                                                     ignore_mismatched_sizes=True)

    def forward(self, x):
        """
        Args:
            x (list): List of spectrograms. If list is not given, the spectrogram is converted to a list.

        Returns:
            torch.Tensor: Output tensor (logits) of shape (batch_size, `no_classes`).
        """
        if not isinstance(x, list):
            # make every file in a batch (length, ) instead of (1, length)
            x = x.squeeze(1).tolist()
        features = self.featurizer(x, return_tensors="pt", sampling_rate=self.featurizer.sampling_rate)
        features = features.to(self.device)
        return self.model(features.input_values).logits

    def get_cls_named_parameters(self):
        """
        Returns:
            list: List of named parameters of the classifier.
        """
        return [n for n, _ in self.model.classifier.named_parameters(prefix="model.classifier")]