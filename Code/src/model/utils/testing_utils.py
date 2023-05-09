import numpy as np
import torch
from sklearn.metrics import hamming_loss

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchmetrics.classification import MultilabelAccuracy

NO_CLASSES = 11

def calculate_metrics(pred, target, threshold=0.5, no_classes=NO_CLASSES):
    pred = np.array(pred > threshold, dtype=int)
    micro_accuracy = MultilabelAccuracy(no_classes, threshold, average='micro')
    macro_accuracy = MultilabelAccuracy(no_classes, threshold, average='macro')

    return {
        'micro_accuracy': micro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'macro_accuracy': macro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'exact_match_accuracy': accuracy_score(target, pred),
        'micro_precision': precision_score(target, pred, average='micro', zero_division=0),
        'macro_precision': precision_score(target, pred, average='macro', zero_division=0),
        'macro_recall': recall_score(target, pred, average='macro', zero_division=0),
        'macro_f1': f1_score(target, pred, average='macro', zero_division=0),
        'micro_recall': recall_score(target, pred, average='micro', zero_division=0),
        'micro_f1': f1_score(target, pred, average='micro', zero_division=0),
        'samples_precision': precision_score(target, pred, average='samples', zero_division=0),
        'samples_recall': recall_score(target, pred, average='samples', zero_division=0),
        'samples_f1': f1_score(target, pred, average='samples', zero_division=0),
        'hamming_score': 1-hamming_loss(target, pred)
    }