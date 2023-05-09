import torch

from src.model.models.abstract_model import AbstractModel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


class AST(AbstractModel):
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
        if not isinstance(x, list):
            # make every file in a batch (length, ) instead of (1, length)
            x = x.squeeze(1).tolist()
        features = self.featurizer(x, return_tensors="pt", sampling_rate=self.featurizer.sampling_rate)
        features = features.to(self.device)
        return self.model(features.input_values).logits

    def get_cls_named_parameters(self):
        return [n for n, _ in self.model.classifier.named_parameters(prefix="model.classifier")]