from typing import Any

import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
from src.models.abstract_model import AbstractModel
from src.utils.utils import calculate_metrics
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class AudioLitModule(pl.LightningModule):
    def __init__(self, net: AbstractModel,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_warmup_percentage: float,
                 no_classes: int,
                 threshold_value: int,
                 aggregation_function: str,
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
        val_loss = self.eval_criterion(torch.tensor(np.array(self.eval_outputs_list)).float(), torch.tensor(np.array(self.eval_targets_list)).float())
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log_dict(dictionary=result_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.eval_targets_list.clear()
        self.eval_outputs_list.clear()

    def test_step(self, batch: Any, batch_idx: int):
        self.eval_step(batch)

    def eval_step(self, batch: Any):
        features, targets, lengths = batch
        for i in range (len(features)):
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
            scheduler = self.hparams.scheduler(optimizer=optimizer, num_warmup_steps=self.hparams.scheduler_warmup_percentage * num_steps,
                                               num_training_steps=num_steps)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step"
                    },
                }
        return {"optimizer": optimizer}


class AudioLitModuleSeparateOptimizers(AudioLitModule):
    def __init__(self, net: AbstractModel,
                 optimizer_base: torch.optim.Optimizer,
                 optimizer_classifier: torch.optim.Optimizer,
                 scheduler_base: torch.optim.lr_scheduler,
                 scheduler_classifier: torch.optim.lr_scheduler,
                 scheduler_warmup_percentage: float,
                 no_classes: int,
                 threshold_value: int,
                 aggregation_function: str,
                 gradient_accumulation_steps: int,
                 apply_gradient_clipping: bool,
                 ):
        super().__init__(net, optimizer_base, scheduler_base, scheduler_warmup_percentage, no_classes, threshold_value, aggregation_function)

        self.automatic_optimization = False
        self.save_hyperparameters(logger=False)

    def configure_optimizers(self):
        cls_named_params = self.net.get_cls_named_parameters()
        params_base, params_classifier = [], []
        for n, p in self.net.named_parameters():
            if n in cls_named_params:
                params_classifier.append(p)
            else:
                params_base.append(p)
        assert len(params_classifier) > 0, "No classifier parameters found, check the named parameters returned by the model."
        optimizer_base = self.hparams.optimizer_base(params=params_base)
        optimizer_classifier = self.hparams.optimizer_classifier(params=params_classifier)
        num_steps = self.trainer.estimated_stepping_batches / self.hparams.gradient_accumulation_steps
        if self.hparams.scheduler_base is not None:
            scheduler_base = self.hparams.scheduler_base(optimizer=optimizer_base, num_warmup_steps=self.hparams.scheduler_warmup_percentage * num_steps, num_training_steps=num_steps)
        if self.hparams.scheduler_classifier is not None:
            scheduler_classifier = self.hparams.scheduler_classifier(optimizer=optimizer_classifier, num_warmup_steps=self.hparams.scheduler_warmup_percentage * num_steps, num_training_steps=num_steps)
        if self.hparams.scheduler_base is not None and self.hparams.scheduler_classifier is not None: # both schedulers
            return [optimizer_base, optimizer_classifier], [scheduler_base, scheduler_classifier]
        elif self.hparams.scheduler_base is not None: # only base scheduler
            return [optimizer_base, optimizer_classifier], [scheduler_base]
        elif self.hparams.scheduler_classifier is not None: # only classifier scheduler
            return [optimizer_base, optimizer_classifier], [scheduler_classifier]
        else: # no schedulers
            return [optimizer_base, optimizer_classifier]
        
    def training_step(self, batch: Any, batch_idx: int):
        _, _, loss = self.model_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.manual_backward(loss)
        if self.hparams.apply_gradient_clipping:
            for optimizer in self.optimizers():
                self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
        if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
            optimizers, lr_schedulers = self.optimizers(), self.lr_schedulers()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
