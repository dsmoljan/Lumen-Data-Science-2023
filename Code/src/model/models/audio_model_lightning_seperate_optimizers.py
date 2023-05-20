from typing import Any

import torch
from src.model.models.abstract_model import AbstractModel
from src.model.models.audio_model_lightning import AudioLitModule


class AudioLitModuleSeparateOptimizers(AudioLitModule):
    """
    An implementation of the AbstractModel class which allows the trainig loop to work with two
     seperate optimizers and schedulers. Benefical for large pretrained models whose pretrained-layers require
     updates with a small learning rate, while the head of the model requires a large learning rate.

    Args:
        net (AbstractModel): Model component. \\
        optimizer_base (torch.optim.Optimizer): Optimizer for all the parameters except the classifier. \\
        optimizer_classifier (torch.optim.Optimizer): Optimizer for the classifier. \\
        scheduler_base (torch.optim.lr_scheduler): Scheduler for all the parameters except the classifier. \\
        scheduler_classifier (torch.optim.lr_scheduler): Scheduler for the classifier. \\
        scheduler_warmup_percentage (float): Percentage of the total number of steps to be used for the scheduler warmup. \\
        no_classes (int): Number of classes in the dataset. \\
        threshold_value (int): Threshold value for chossing the positive labels. \\
        aggregation_function (str): Aggregation function to be used for the predictions. \\
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before backpropagation. \\
        apply_gradient_clipping (bool): Whether to apply gradient clipping or not.
    """
    def __init__(self, net: AbstractModel = None,
                 optimizer_base: torch.optim.Optimizer = None,
                 optimizer_classifier: torch.optim.Optimizer = None,
                 scheduler_base: torch.optim.lr_scheduler = None,
                 scheduler_classifier: torch.optim.lr_scheduler = None,
                 scheduler_warmup_percentage: float = None,
                 no_classes: int = 11,
                 threshold_value: int = 0.5,
                 aggregation_function: str = "S2",
                 gradient_accumulation_steps: int = 1,
                 apply_gradient_clipping: bool = False,
                 ):
        super().__init__(net, optimizer_base, scheduler_base, scheduler_warmup_percentage, no_classes, threshold_value,
                         aggregation_function)

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
        assert len(
            params_classifier) > 0, "No classifier parameters found, check the named parameters returned by the model."
        optimizer_base = self.hparams.optimizer_base(params=params_base)
        optimizer_classifier = self.hparams.optimizer_classifier(params=params_classifier)
        num_steps = int(self.trainer.estimated_stepping_batches / self.hparams.gradient_accumulation_steps)
        if self.hparams.scheduler_base is not None:
            scheduler_base = self.hparams.scheduler_base(optimizer=optimizer_base, num_warmup_steps=int(
                self.hparams.scheduler_warmup_percentage * num_steps), num_training_steps=num_steps)
        if self.hparams.scheduler_classifier is not None:
            scheduler_classifier = self.hparams.scheduler_classifier(optimizer=optimizer_classifier,
                                                                     num_warmup_steps=int(
                                                                         self.hparams.scheduler_warmup_percentage * num_steps),
                                                                     num_training_steps=num_steps)
        if self.hparams.scheduler_base is not None and self.hparams.scheduler_classifier is not None:  # both schedulers
            return [optimizer_base, optimizer_classifier], [scheduler_base, scheduler_classifier]
        elif self.hparams.scheduler_base is not None:  # only base scheduler
            return [optimizer_base, optimizer_classifier], [scheduler_base]
        elif self.hparams.scheduler_classifier is not None:  # only classifier scheduler
            return [optimizer_base, optimizer_classifier], [scheduler_classifier]
        else:  # no schedulers
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
            for optimizer, lr_scheduler in zip(optimizers, lr_schedulers):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        return loss