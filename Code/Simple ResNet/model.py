import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

import utils
from data_utils import data_utils as du
from data_utils.IRMAS_dataloader import IRMASDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NO_CLASSES = 11
# kasnije eventualno izabrati bolji način određivanja thresholda
THRESHOLD_VALUE = 0.18

DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

tensorboard_loc = './tensorboard_results/first_run'

class classifierModel(object):
    def __init__(self, args):
        self.model = models.resnet50(pretrained=True)
        self.args = args

        num_ftrs_in = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs_in, NO_CLASSES))

        utils.print_networks([self.model], ['ResNet'])

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.args.weight_decay)

        if not os.path.isdir(self.args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3,
                                                                    verbose=True)

        # ovako čudan kod za loadanje je zbog greske s optimizerom https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        try:
            ckpt = utils.load_checkpoint('%s/classification_model.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state'])
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.args.weight_decay)
            self.optimizer.load_state_dict(ckpt['model_optimizer'])
            self.scheduler.load_state_dict(ckpt['model_scheduler'])
            self.best_macro_f1 = ckpt['best_macro_f1']
        except Exception as e:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_macro_f1 = -100

        self.model = self.model.to(self.device)
        self.activation = nn.Sigmoid()
        self.writer_classifier = SummaryWriter(tensorboard_loc + '_classifier')


    def forward(self, x):
        return self.activation(self.model(x))

    def train(self):
        train_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='train', audio_augmentation=True, spectogram_augmentation=True, sr=44100, return_type='image')
        val_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='val', audio_augmentation=False, spectogram_augmentation=False, sr=44100, return_type='image')

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        self.model.train()

        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_train_loss = 0
            self.model.train()
            count = 0
            # tqdm(test_loader, desc='Testing', leave=False, total=len(test_loader))
            for imgs, targets in tqdm(train_loader, desc='Training', leave=False, total=len(train_loader)):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                # torch BCELoss doesn't apply an activation, so we need to apply one before calling the loss function
                # in contrast, CE loss included a softmax activation in it, so if we were using it, we wouldn't apply an activation before
                outputs = self.forward(imgs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_train_loss += loss.item()

                count += 1

            epoch_train_loss /= count
            print("Epoch: {}", epoch)
            print("Train loss: {}", epoch_train_loss)
            self.writer_classifier.add_scalars('Train metrics', {'BCE Loss ': epoch_train_loss,
                                                                 'Learning_rate': self.optimizer.param_groups[0]["lr"]}, epoch)

            if (self.args.use_validation):
                print("Evaluating model on val set")
                self.eval(epoch, val_loader)
            else:
                print("Updating checkpoint - not using validation, so saving after every epoch")
                torch.save({'epoch': epoch + 1,
                                       'model_state': self.model.state_dict(),
                                       'model_optimizer': self.optimizer.state_dict(),
                                       'best_macro_f1': self.best_macro_f1},
                                      '%s/classification_model.ckpt' % self.args.checkpoint_dir)

    def eval(self, epoch, val_loader):
        self.model.eval()

        outputs_list = []
        targets_list = []

        with torch.no_grad():
            val_loss = 0
            counter = 0
            for i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.forward(imgs)

                outputs_list.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
                val_loss += self.criterion(outputs, targets).item()
                counter += 1

            val_loss /= counter
            result = calculate_metrics(np.array(outputs_list), np.array(targets_list))
            self.scheduler.step(result['macro/f1'])
            print("Validation results")
            print("Epoch:{:2d}: "
                  "Micro f1: {:.3f} "
                  "Macro f1: {:.3f} "
                  "Samples f1: {:.3f},"
                  "Accuracy : {:.3f}".format(epoch,
                                              result['micro/f1'],
                                              result['macro/f1'],
                                              result['samples/f1'],
                                              result['accuracy_score']))

            self.writer_classifier.add_scalars('Val Metrics', {'CE Loss': val_loss, 'Accuracy': result['accuracy_score'],
                                                               'Micro F1': result['micro/f1'],
                                                               'Macro F1': result['macro/f1'],
                                                               'Samples F1': result['samples/f1']},epoch)

            if (result['macro/f1'] > self.best_macro_f1):
                self.best_macro_f1 = result['macro/f1']
                torch.save({'epoch': epoch + 1,
                            'model_state': self.model.state_dict(),
                            'model_scheduler': self.scheduler.state_dict(),
                            'model_optimizer': self.optimizer.state_dict(),
                            'best_macro_f1': self.best_macro_f1},
                           '%s/classification_model.ckpt' % self.args.checkpoint_dir)

    def test(self):
        self.model.eval()

        outputs_list = []
        targets_list = []

        test_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='test', audio_augmentation=False, spectogram_augmentation=False, sr=44100, return_type='image', use_window=True, window_size=3)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True, collate_fn=du.collate_fn_windows)

        with torch.no_grad():
            # features = list of all examples in the batch. One example = multiple audio windows
            for features, targets, lengths in tqdm(test_loader, desc='Testing', leave=False, total=len(test_loader)):
                for i in range(len(features)):
                    imgs = features[i].to(self.device)[0:lengths[i]]
                    target = targets[i]
                    outputs = self.forward(imgs).cpu().numpy()

                    if (self.args.aggregation_function == "S2"):
                        outputs_sum = np.sum(outputs, axis=0)
                        max_val = np.max(outputs_sum)
                        outputs_sum /= max_val
                    else:
                        outputs_sum = np.mean(outputs, axis=0)

                    outputs_list.extend(np.expand_dims(outputs_sum, axis=0))
                    targets_list.extend(target.unsqueeze(dim=0).cpu().numpy())

            result = calculate_metrics(np.array(outputs_list), np.array(targets_list))
            print("Test results")
            print("Micro f1: {:.3f} "
                  "Macro f1: {:.3f} "
                  "Samples f1: {:.3f},"
                  "Accuracy : {:.3f}".format(result['micro/f1'],
                                             result['macro/f1'],
                                             result['samples/f1'],
                                             result['accuracy_score']))

# koristi micro F1
def calculate_metrics(pred, target, threshold=THRESHOLD_VALUE):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            'accuracy_score': accuracy_score(y_true=target, y_pred=pred)
            }



