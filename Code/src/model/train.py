from typing import List

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from src.model.data_utils.audio_dataset import AudioDataset
from src.model.utils import instantiators, logging_utils, utils
from torch.utils.data import DataLoader

# position yourself in the directory above src, and run "python -m src.train"

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


def train(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg (DictConfig): configuration object from Hydra
    """
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating train dataset <{cfg.data.train_dataset._target_}>")
    train_dataset: AudioDataset = hydra.utils.instantiate(cfg.data.train_dataset)

    log.info(f"Instantiating val dataset <{cfg.data.val_dataset._target_}>")
    val_dataset: AudioDataset = hydra.utils.instantiate(cfg.data.val_dataset)

    train_collate_fn = hydra.utils.get_method(
        cfg.data.train_dataloader.collate_fn) if cfg.data.train_dataloader.collate_fn != "None" else None
    val_collate_fn = hydra.utils.get_method(
        cfg.data.val_dataloader.collate_fn) if cfg.data.val_dataloader.collate_fn != "None" else None

    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=cfg.data.train_dataloader.batch_size,
                                              shuffle=True, drop_last=True, collate_fn=train_collate_fn)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=cfg.data.val_dataloader.batch_size, shuffle=False,
                                            drop_last=True, collate_fn=val_collate_fn)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        logging_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                    ckpt_path=cfg.get("ckpt_path"))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
