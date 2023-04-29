import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


class AST(nn.Module):
    def __init__(self, no_classes, mean, std, model_name_or_path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=no_classes, return_dict=True)
        self.featurizer = AutoFeatureExtractor.from_pretrained(model_name_or_path, mean=mean, std=std)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, config=self.config, ignore_mismatched_sizes=True)

    def forward(self, x):
        # if not type list, convert to list
        if not isinstance(x, list):
            x = x.squeeze(1).tolist()
        features = self.featurizer(x, return_tensors="pt", sampling_rate=16000)
        return self.model(features.input_values).logits
