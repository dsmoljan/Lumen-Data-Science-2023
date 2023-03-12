import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

import utils
from data_utils.IRMAS_dataloader import IRMASDataset

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NO_CLASSES = 11
# kasnije eventualno izabrati bolji način određivanja thresholda
THRESHOLD_VALUE = 0.5

DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

tensorboard_loc = './tensorboard_results/first_run'

# ovo je multi-label classification problem!
# i u skladu s tim treba prilagoditi metrike
# https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/ ---> ovo ti je primjer implementacije!
class classifierModel(object):
    def __init__(self, args):
        self.model = models.resnet50(pretrained=True)
        self.args = args

        # veličina ulaznog tenzora u FC sloj
        num_ftrs_in = self.model.fc.in_features

        # podešavamo head modela - izlazni sloj mreže
        # točnije, resetiramo njegove težine, te podešavamo izlaz -> broj klasa
        # ali to ne znači da se ostale težine modela ne ažuriraju, ažuriraju se!
        self.model.fc = nn.Linear(num_ftrs_in, NO_CLASSES)

        utils.print_networks([self.model], ['ResNet'])

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        if not os.path.isdir(self.args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ovako čudan kod za loadanje je zbog greske s optimizerom https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        try:
            ckpt = utils.load_checkpoint('%s/classification_model.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state'])
            self.model = self.model.to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
            self.optimizer.load_state_dict(ckpt['model_optimizer'])
            self.best_acc = ckpt['best_acc']
        except Exception as e:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_acc = -100

        self.model = self.model.to(self.device)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.model(x))

    def train(self):
        train_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=128, name='train', audio_augmentation=True, spectogram_augmentation=True, sr=44100, return_type='image')
        val_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=128, name='val', audio_augmentation=True, spectogram_augmentation=True, sr=44100, return_type='image')

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        self.model.train()
        count = 0

        epoch_train_loss = 0
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(imgs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                print("Epoch: (%3d) (%5d/%5d) | Crossentropy Loss:%.2e" %
                      (epoch, i + 1, len(train_loader), loss.item()))

                epoch_train_loss += loss.item()

                count += 1

            epoch_train_loss /= count
            print("Epoch: {}", epoch)
            print("Train loss: {}", epoch_train_loss)

def calculate_metrics(pred, target, threshold=0.5):
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
            }


