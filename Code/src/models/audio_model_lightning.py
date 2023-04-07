import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric

from src.data_utils import data_utils as du
from src.data_utils.IRMAS_dataloader import IRMASDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAccuracy
from tqdm import tqdm

import pytorch_lightning as pl

from src.utils.utils import calculate_metrics

NO_CLASSES = 11
THRESHOLD_VALUE = 0.5

class AudioLitModule(pl.LightningModule):
    def __init__(self, net: nn.Module, args):
        super.__init__()

        self.net = net

        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()

        self.args = args
        self.train_macro_acc = MultilabelAccuracy(NO_CLASSES, THRESHOLD_VALUE, average='macro')
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.best_val_metric = MaxMetric()
        self.val_macro_f1 = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x))

    # by default lightning executes validation step sanity checks before training starts,
    # so it's worth to make sure validation metrics don't store results from these checks
    def on_train_start(self) -> None:
        self.best_val_metric.reset()

    def model_step(self, batch: Any):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim = 1)
        return preds, targets, loss

    def training_step(self, batch: Any, batch_idx: int):
        preds, targets, loss = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # per Ligthning module requirements, we return loss so Lightning can perform backprop and optimizer steps
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        preds, targets, loss = self.model_step(batch)

        self.val_loss(loss)
        result_dict = calculate_metrics(np.array(preds), np.array(targets))
        # Lightning should automatically aggregate and calculate mean of every metric in the dict
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.val_macro_f1.update(result_dict["macro_f1"])
        return result_dict

    def on_validation_end(self):
        val_epoch_macro_f1 = self.val_macro_f1.compute()
        self.best_val_metric(val_epoch_macro_f1)


    def test_step(self, batch: Any, batch_idx: int):
        preds, targets, loss = self.model_step(batch)

        self.test_loss(loss)
        result_dict = calculate_metrics(np.array(preds), np.array(targets))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Lightning should automatically aggregate and calculate mean of every metric in the dict
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss", # TODO -> ovo prebaciti u macro_f1
                    "interval": "epoch",
                    "frequency": 1,
                },
            }







