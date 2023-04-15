from typing import Any

import numpy as np
import pyrootutils
import torch
from torchmetrics import MaxMetric, MeanMetric

from torch import nn
from torchmetrics.classification import MultilabelAccuracy

import pytorch_lightning as pl

from src.utils.utils import calculate_metrics

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class AudioLitModule(pl.LightningModule):
    def __init__(self, net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 no_classes: int,
                 threshold_value: int,
                 aggregation_function: str
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()

        self.train_macro_acc = MultilabelAccuracy(no_classes, threshold_value, average='macro')
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
        #preds = torch.argmax(logits, dim = 1)
        return logits, targets, loss

    def training_step(self, batch: Any, batch_idx: int):
        logits, targets, loss = self.model_step(batch)

        self.train_loss(loss)
        self.train_macro_acc(logits, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # per Ligthning module requirements, we return loss so Lightning can perform backprop and optimizer steps
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        logits, targets, loss = self.model_step(batch)

        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        result_dict = calculate_metrics(logits.cpu().numpy(), targets.cpu().numpy())
        # Lightning should automatically aggregate and calculate mean of every metric in the dict
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_macro_f1.update(result_dict["macro_f1"])

    def on_validation_end(self):
        val_epoch_macro_f1 = self.val_macro_f1.compute()
        self.best_val_metric(val_epoch_macro_f1)


    def test_step(self, batch: Any, batch_idx: int):
        features, targets, lengths = batch
        for i in range (len(features)):
            examples = features[i][0:lengths[i]]
            target = targets[i]
            logits = self.forward(examples)

            if self.hparams.aggregation_function == "S2":
                outputs_sum = np.sum(logits.cpu().numpy(), axis=0)
                max_val = np.max(outputs_sum)
                outputs_sum /= max_val
            else:
                outputs_sum = np.mean(logits, axis=0)

            #self.test_loss(loss)
            result_dict = calculate_metrics(np.expand_dims(np.array(outputs_sum), axis=0), np.array(target.unsqueeze(dim=0).cpu().numpy()))
            #self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            # Lightning should automatically aggregate and calculate mean of every metric in the dict
            self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss", # TODO -> ovo prebaciti u macro_f1
                        "interval": "epoch",
                        "strict": False,
                        "frequency": 1,
                    },
                }
        return {"optimizer": optimizer}







