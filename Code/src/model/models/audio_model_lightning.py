from typing import Any

import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
from src.model.models.abstract_model import AbstractModel
from src.model.utils.testing_utils import calculate_metrics
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class AudioLitModule(pl.LightningModule):
    """
    Base implementation of the AbstractModel class. Contains all the code common for all models,
    such as training step, optimizer config, evaluation loop etc. The specific architecture to be used
    is taken as a parameter. Does NOT implement get_cls_named_parameters method from AbstractModel, as
    it does not support working with multiple optimizers and schedulers.

    Args:
        net (AbstractModel): Model component. \\
        optimizer (torch.optim.Optimizer): Optimizer. \\
        scheduler (torch.optim.lr_scheduler): Scheduler. \\
        scheduler_warmup_percentage (float): Percentage of the total number of steps to be used for the scheduler warmup. \\
        no_classes (int): Number of classes in the dataset. \\
        threshold_value (int): Threshold value for chossing the positive labels. \\
        aggregation_function (str): Aggregation function to be used for the predictions.
    """
    def __init__(self, net: AbstractModel = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 scheduler_warmup_percentage: float = None,
                 no_classes: int = 11,
                 threshold_value: int = 0.5,
                 aggregation_function: str = "S2",
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_criterion = nn.BCEWithLogitsLoss()
        self.eval_criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()

        self.eval_outputs_list = []
        self.eval_targets_list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # by default lightning executes validation step sanity checks before training starts,
    # so it's worth to make sure validation metrics don't store results from these checks

    def model_step(self, batch: Any):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.train_criterion(logits, targets)
        return logits, targets, loss

    def training_step(self, batch: Any, batch_idx: int):
        _, _, loss = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.eval_step(batch)

    def on_validation_epoch_end(self) -> None:
        result_dict = calculate_metrics(np.array(self.eval_outputs_list), np.array(self.eval_targets_list))
        # logger = False as this log is only so we can use the scheduler, the metric is already logged to wandb
        # by logging the entire results dict
        val_loss = self.eval_criterion(torch.tensor(np.array(self.eval_outputs_list)).float(),
                                       torch.tensor(np.array(self.eval_targets_list)).float())
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.eval_targets_list.clear()
        self.eval_outputs_list.clear()

    def test_step(self, batch: Any, batch_idx: int):
        self.eval_step(batch)

    def eval_step(self, batch: Any):
        features, targets, lengths = batch
        for i in range(len(features)):
            examples = features[i][0:lengths[i]]
            target = targets[i]
            predictions = self.activation(self.forward(examples))

            if self.hparams.aggregation_function == "S2":
                outputs_sum = np.sum(predictions.cpu().numpy(), axis=0)
                max_val = np.max(outputs_sum)
                outputs_sum /= max_val
            else:
                outputs_sum = np.mean(predictions.cpu().numpy(), axis=0)

            self.eval_outputs_list.extend(np.expand_dims(outputs_sum, axis=0))
            self.eval_targets_list.extend(target.unsqueeze(dim=0).cpu().numpy())

    def on_test_epoch_end(self) -> None:
        result_dict = calculate_metrics(np.array(self.eval_outputs_list), np.array(self.eval_targets_list))
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.eval_targets_list.clear()
        self.eval_outputs_list.clear()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        num_steps = self.trainer.estimated_stepping_batches
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer,
                                               num_warmup_steps=self.hparams.scheduler_warmup_percentage * num_steps,
                                               num_training_steps=num_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }
        return {"optimizer": optimizer}