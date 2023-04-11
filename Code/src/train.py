from typing import List

import hydra
import pytorch_lightning as pl
import pyrootutils
import torch
from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, Logger
from torch.utils.data import DataLoader

from src.data_utils.IRMAS_dataloader import IRMASDataset
from src.models.audio_model_lightning import AudioLitModule
from src.models.components.cnn_spectogram_net import CNNSpectogramNet

from omegaconf import DictConfig

from src.utils import utils, instantiators, logging_utils

data_root_path = "../../../Dataset/"
batch_size = 8
DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

# jedan siguran način za pokrenuti ovo
# pozicioniraš se u direktorij iznad src, te pokreneš "python -m src.train"
# zasad to radi, a čini se da ima neki library pyrootutils koji koriste u onom example projektu, pa kasnije možeš
# probati preko njega

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)

def train(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating train dataloader <{cfg.data.train_data._target_}>")
    train_dataloader: DataLoader = hydra.utils.instantiate(cfg.data.train_data)

    log.info(f"Instantiating val dataloader <{cfg.data.val_data._target_}>")
    val_dataloader: DataLoader = hydra.utils.instantiate(cfg.data.val_data)

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     mode="min",
    #     filename="resnet_spectogram_model_{epoch:02d}_{val_loss:.3f}",
    #     save_top_k=2,
    #     save_last=True,  # always save the last checkpoint
    # )
    #
    # wandb_logger = WandbLogger(project="lumen-test")
    # trainer = pl.Trainer(max_epochs=20, check_val_every_n_epoch=1, callbacks=[checkpoint_callback], logger=wandb_logger)
    # train_set = IRMASDataset(data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='train',
    #                          audio_augmentation=True, spectogram_augmentation=True, sr=44100, return_type='spectogram')
    # val_set = IRMASDataset(data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='val',
    #                        audio_augmentation=False, spectogram_augmentation=False, sr=44100, return_type='spectogram')
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    #
    # base_model = CNNSpectogramNet()
    # model = AudioLitModule(net=base_model)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train(cfg)


if __name__ == "__main__":
    main()
