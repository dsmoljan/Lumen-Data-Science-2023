from argparse import ArgumentParser

import torch

from models.cnn_audio_model import Conv1DModel
from src.models.cnn_mfcc_model import Conv2DMFCCModel
from src.models.resnet_spectogram_model import ResnetSpectogramModel


# lr = 0.0002 radi drastično bolje nego lr 0.001
def get_args():
    parser = ArgumentParser(description='IRMAS ResNet model PyTorch')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=20)
    parser.add_argument('--crop_width', type=int, default=40)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--use_validation', type=bool, default=True)
    parser.add_argument('--model', type=str, default='1d_conv_audio')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--validation_dir', type=str, default='./val_results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_root_path', type=str, default='../../../Dataset/')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--n_mfcc', type=int, default=40)
    parser.add_argument('--n_mels', type=int, default=256)
    parser.add_argument('--aggregation_function', type=str, default="S2")
    args = parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_args()
    model = None
    if args.model == '1d_conv_audio':
        model = Conv1DModel(args)
    elif args.model == '2d_conv_mfcc':
        model = Conv2DMFCCModel(args)
    elif args.model == 'resnet_spectograms':
        model = ResnetSpectogramModel(args)
    else:
        raise NotImplemented(f"The model with name {args.model} is not implemented")
    if args.training:
        print(f"Training {args.model} model")
        model.training_loop()
    if args.testing:
        print(f"Testing {args.model} model")
        model.testing_loop()
