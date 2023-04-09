import pyrootutils
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data_utils.IRMAS_dataloader import IRMASDataset
from src.data_utils.data_utils import collate_fn_windows
from src.models.audio_model_lightning import AudioLitModule
from src.models.components.cnn_spectogram_net import CNNSpectogramNet

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

data_root_path = "../../../Dataset/"
ckpt_path = "../checkpoints/last.ckpt"
batch_size = 8
DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

def main():
    base_model = CNNSpectogramNet()
    model = AudioLitModule(net=base_model)
    wandb_logger = WandbLogger(project="lumen-test")
    trainer = pl.Trainer(logger=wandb_logger)

    test_set = IRMASDataset(data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='test',
                           audio_augmentation=False, spectogram_augmentation=False, sr=44100, return_type='spectogram', use_window=True, window_size=3)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_windows)

    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()

