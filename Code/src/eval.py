from typing import List

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer, Callback
from torch.utils.data import DataLoader

from src.data_utils.IRMAS_dataloader import IRMASDataset
from src.data_utils.data_utils import collate_fn_windows
from src.utils import utils, instantiators
from pytorch_lightning.loggers import Logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


def evaluate(cfg: DictConfig):
    assert cfg.ckpt_path

    log.info(f"Instantiating test dataset <{cfg.data.test_dataset._target_}>")
    test_dataset: IRMASDataset = hydra.utils.instantiate(cfg.data.test_dataset)

    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=cfg.data.test_dataloader.batch_size, shuffle=True,
                                             drop_last=True, collate_fn=collate_fn_windows)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "test_dataset": test_dataset,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=cfg.ckpt_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
