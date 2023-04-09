import pytorch_lightning as pl
import pyrootutils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data_utils.IRMAS_dataloader import IRMASDataset
from src.models.audio_model_lightning import AudioLitModule
from src.models.components.cnn_spectogram_net import CNNSpectogramNet

data_root_path = "../../../Dataset/"
batch_size = 8
DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

# jedan siguran način za pokrenuti ovo
# pozicioniraš se u direktorij iznad src, te pokreneš "python -m src.train"
# zasad to radi, a čini se da ima neki library pyrootutils koji koriste u onom example projektu, pa kasnije možeš
# probati preko njega

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def main():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="resnet_spectogram_model_{epoch:02d}_{val_loss:.3f}",
        save_top_k=2,
        save_last=True, # always save the last checkpoint
    )

    wandb_logger = WandbLogger(project="lumen-test")
    trainer = pl.Trainer(max_epochs=20, check_val_every_n_epoch=1, callbacks=[checkpoint_callback], logger=wandb_logger)
    train_set = IRMASDataset(data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='train',
                                 audio_augmentation=True, spectogram_augmentation=True, sr=44100, return_type='spectogram')
    val_set = IRMASDataset(data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='val',
                           audio_augmentation=False, spectogram_augmentation=False, sr=44100, return_type='spectogram')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

    base_model = CNNSpectogramNet()
    model = AudioLitModule(net=base_model)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()