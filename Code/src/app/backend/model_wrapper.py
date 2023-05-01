from typing import List

import numpy as np
import torch
import json

from torch import nn

from src.app.backend.preprocessing import preprocess_spectogram

BASE_MODEL_CHECKPOINT = "./checkpoints/resnet_model_window_3s.pt"

class_mappings = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}

class_labels = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

WINDOW_SIZE = 3

SR = 44100
N_MFCC = 13
N_MELS = 256
THRESHOLD = 0.5

model_type = "resnet_spectograms"

class ModelWrapper():
    """
    A wrapper class that helps with loading and getting predictions from
    models saved as Torchscript files.
    """
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model = torch.jit.load(BASE_MODEL_CHECKPOINT)
        self.base_model.to(self.device)
        self.activation = nn.Sigmoid()

    def predict(self, audio_file, sr):
        with torch.no_grad():
            audio_windows_tensor = preprocess_spectogram(audio_file, sr=sr, n_mels=N_MELS, height=N_MELS, width=N_MELS, window_size=WINDOW_SIZE)
            audio_windows_tensor = audio_windows_tensor.to(self.device)
            predictions = self.activation(self.base_model(audio_windows_tensor))

            # S2 aggregation function
            outputs_sum = np.sum(predictions.cpu().numpy(), axis=0)
            max_val = np.max(outputs_sum)
            outputs_sum /= max_val

            final_predictions = np.array(outputs_sum > THRESHOLD, dtype=int)

            class_dict = {class_labels[i]: int(final_predictions[i]) for i in range(len(class_labels))}
            class_json = json.dumps(class_dict)
            return class_dict

    # imaj na umu da ako radiš s modelom treniranim na audiosetom, onda moraš normalizirati koristeći
    # njegov mean i std. devijaciju
    def preprocess(self, audio_file: np.ndarray, sr:int) -> List[torch.Tensor]:
        if model_type == "resent_spectograms":
            return preprocess_spectogram(audio_file=audio_file, sr=sr, n_mels=N_MELS, height=N_MELS, width=N_MELS, window_size=WINDOW_SIZE)
        else:
            raise NotImplementedError("Other model types are not supported")


def aggregate_S2(probabilities):
    pass

