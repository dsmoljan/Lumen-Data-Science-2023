from argparse import ArgumentParser
import model as md

# lr = 0.0002 radi drastično bolje nego lr 0.001
def get_args():
    parser = ArgumentParser(description='IRMAS ResNet model PyTorch')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=20)
    parser.add_argument('--crop_width', type=int, default=40)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--use_validation', type=bool, default=True)
    #parser.add_argument('--model', type=str, default='supervised_model')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--validation_dir', type=str, default='./val_results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_root_path', type=str, default='../../../Dataset/')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    args = parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_args()
    if args.training:
        print("Training classifier model")
        model = md.classifierModel(args)
        model.train()
    if args.testing:
        print("Testing")
        model = md.classifierModel(args)
        model.test()
