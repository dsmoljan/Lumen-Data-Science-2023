from typing import Any

import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
from src.utils.utils import calculate_metrics
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MultilabelAccuracy

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

        self.eval_outputs_list = []
        self.eval_targets_list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x))

    # by default lightning executes validation step sanity checks before training starts,
    # so it's worth to make sure validation metrics don't store results from these checks

    def model_step(self, batch: Any):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        return logits, targets, loss

    def training_step(self, batch: Any, batch_idx: int):
        _, _, loss = self.model_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # per Ligthning module requirements, we return loss so Lightning can perform backprop and optimizer steps
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        logits, targets, loss = self.model_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.eval_outputs_list.extend(logits.cpu().numpy())
        self.eval_targets_list.extend(targets.cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        result_dict = calculate_metrics(np.array(self.eval_outputs_list), np.array(self.eval_targets_list))
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.eval_targets_list.clear()
        self.eval_outputs_list.clear()

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

            self.eval_outputs_list.extend(np.expand_dims(outputs_sum, axis=0))
            self.eval_targets_list.extend(target.unsqueeze(dim=0).cpu().numpy())

    def on_test_epoch_end(self) -> None:
        result_dict = calculate_metrics(np.array(self.eval_outputs_list), np.array(self.eval_targets_list))
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.eval_targets_list.clear()
        self.eval_outputs_list.clear()


    # TODO: vrati scheduler!
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
        # TODO: ovo kako je trenutno implementirano ne podrzava druge schedulere koji rade step svaki step umjesto svaku epohu (linear scheduler npr)
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
