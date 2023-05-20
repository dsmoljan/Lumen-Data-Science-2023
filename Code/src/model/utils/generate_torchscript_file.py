import torch

from src.model.models.audio_model_lightning import AudioLitModule
from src.model.models.components.cnn_spectogram_net import CNNSpectogramNet

CHECKPOINT_PATH = "epoch_000.ckpt"


net = CNNSpectogramNet()
def generate_torchscript_file():
    model = AudioLitModule.load_from_checkpoint(CHECKPOINT_PATH, net=net)
    script = model.to_torchscript()
    torch.jit.save(script, "model.pt")


if __name__ == "__main__":
    generate_torchscript_file()