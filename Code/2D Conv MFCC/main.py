from argparse import ArgumentParser

import model as md
from model import test, train
from torchsummary import summary


def get_args():
    parser = ArgumentParser(description='IRMAS Conv 1D Model on Raw Audio Data - PyTorch')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--use_validation', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--validation_dir', type=str, default='./val_results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_root_path', type=str, default='../../../Dataset/')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--n_mfcc', type=int, default=13)
    parser.add_argument('--n_mels', type=int, default=256)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.training:
        print("Training classifier model")
        model = md.Conv2DMFCCModel(args)
        #summary(model, (3, 13, 128))
        train(model)
    if args.testing:
        print("Testing")
        model = md.Conv2DMFCCModel(args)
        test(model)