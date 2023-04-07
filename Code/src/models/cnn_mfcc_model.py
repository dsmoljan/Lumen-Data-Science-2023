import os

import numpy as np
import torch
import torch.nn.functional as F
from src.data_utils import data_utils as du
from src.data_utils.IRMAS_dataloader import IRMASDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAccuracy
from tqdm import tqdm

from src.utils.utils import calculate_metrics

# TODO: kad jednom dodaš hydru, ove podatke učitavati iz hydre
NO_CLASSES = 11
THRESHOLD_VALUE = 0.5

DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

tensorboard_loc = './tensorboard_results/2d_conv_mfcc'


# create a model based on 1d convolutional neural network
class Conv2DMFCCModel(nn.Module):
    """
    Model based on 1d convolutional neural network

    Args:
        args: arguments from the command line

    """

    def __init__(self, args):
        super(Conv2DMFCCModel, self).__init__()
        # TODO define the layers
        # the model expects n_mels to be (default) 256 and n_mfcc to be (default) 40 and builds the network accordingly
        # input of shape (batch_size, 1, self.args.n_mfcc, self.args.n_mels)
        self.input_shape = (args.batch_size, 1, args.n_mfcc, args.n_mels)  # batch_size, 1, 40, 256
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4))  # batch_size, 32, 37, 253
        self.bn1 = nn.BatchNorm2d(32)  # batch_size, 32, 37, 253
        self.pool1 = nn.MaxPool2d(2)  # batch_size, 32, 18, 126
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 10))  # batch_size, 64, 17, 117
        self.bn2 = nn.BatchNorm2d(64)  # batch_size, 64, 17, 117
        self.pool2 = nn.MaxPool2d(2)  # batch_size, 64, 8, 58
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 10))  # batch_size, 128, 7, 49
        self.bn3 = nn.BatchNorm2d(128)  # batch_size, 128, 7, 49
        self.pool3 = nn.MaxPool2d(2)  # batch_size, 128, 3, 24
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(2, 10))  # batch_size, 256, 2, 15
        self.bn4 = nn.BatchNorm2d(256)  # batch_size, 256, 2, 15
        self.pool4 = nn.MaxPool2d(2)  # batch_size, 256, 1, 7
        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, NO_CLASSES)

        self.relu = nn.ReLU()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # patience was 3
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3,
                                                                    verbose=True)
        self.activation = nn.Sigmoid()
        self.writer = SummaryWriter(tensorboard_loc)

        try:
            ckpt = torch.load(self.args.checkpoint_dir + '2d_conv_mfcc_classifier.pth')
            self.start_epoch = ckpt['epoch'] + 1
            self.load_state_dict(ckpt['model_state'])
            self.to(self.device)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.optimizer.load_state_dict(ckpt['model_optimizer'])
            self.scheduler.load_state_dict(ckpt['model_scheduler'])
            self.best_val_loss = ckpt['best_val_loss']
            print(" [*] Checkpoint loaded successfully!")
        except Exception as e:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_val_loss = np.inf
        self.to(self.device)

    def forward(self, x):
        x = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool1,
            self.conv2, self.bn2, self.relu, self.pool2,
            self.conv3, self.bn3, self.relu, self.pool3,
            self.conv4, self.bn4, self.relu, self.pool4
        )(x)
        x = F.avg_pool2d(x, x.size()[-2:])
        x = x.view(x.size(0), -1)
        x = nn.Sequential(self.fc1, self.dropout1, self.relu, self.fc2)(x)
        return x

    # training loop
    def training_loop(self):
        train_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=self.args.n_mels,
                                 n_mfcc=self.args.n_mfcc, name='train', audio_augmentation=True, mfcc_augmentation=True,
                                 sr=self.args.sr, return_type='mfcc')
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        micro_accuracy = MultilabelAccuracy(NO_CLASSES, THRESHOLD_VALUE, average='micro').to(self.device)

        for epoch in range(self.start_epoch, self.args.epochs):
            self.train()
            train_loss = 0
            train_acc = 0

            for data, target in tqdm(train_loader, desc='Training', leave=False, total=len(train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(self.activation(output), target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                # micro accuracy requires logit float outputs and applies sigmoid internally
                train_acc += micro_accuracy(output, target)

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            self.writer.add_scalars('Train metrics', {'loss': train_loss, 'micro_accuracy': train_acc,
                                                       'learning_rate': self.optimizer.param_groups[0]["lr"]}, epoch)
            print(f'\nEpoch: {epoch + 1}/{self.args.epochs}')
            print(f'Train Loss: {train_loss:.4f} | Train Micro Acc: {train_acc:.4f}')

            self.eval_loop(epoch)


    # evaluation loop
    def eval_loop(self, epoch):
        val_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=self.args.n_mels,
                               n_mfcc=self.args.n_mfcc, name='val', sr=self.args.sr, return_type='mfcc', use_window=True,
                               window_size=3)

        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                                collate_fn=du.collate_fn_windows)

        with torch.no_grad():
            outputs_list = []
            targets_list = []

            self.eval()
            for features, targets, lengths in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

                current_audio_predictions = np.zeros_like(targets.cpu().numpy(), dtype=np.float32)
                # max_val = np.zeros(targets.shape[0], dtype=np.float32)

                # iterate over the first dimension of the features tensor
                for i in range(features.shape[0]):
                    data = features[i].to(self.device)
                    output = self.activation(self(data)).cpu().numpy()

                    # add output to current_audio_predictions at index i, but take into account only the first n lengths whenre n is given by i-th element of lengths tensor
                    current_audio_predictions[i] = np.sum(output[:lengths[i]], axis=0)

                # divide current_audio_predictions by maximum value of current_audio_predictions at the corresponding index of axis zero to normalize the values
                current_audio_predictions /= np.max(current_audio_predictions, axis=1, keepdims=True)

                outputs_list.extend(current_audio_predictions)
                targets_list.extend(targets.cpu().numpy())

            val_loss = self.criterion(torch.tensor(outputs_list, dtype=torch.float32),
                                       torch.tensor(targets_list, dtype=torch.float32))
            self.scheduler.step(val_loss)
            result = calculate_metrics(np.array(outputs_list), np.array(targets_list))

            print("Validation metrics:")
            print("Loss: {:.3f} | "
                  "Micro f1: {:.3f} | "
                  "Micro Accuracy: {:.3f} | "
                  "Micro Precision: {:.3f} | "
                  "Micro Recall: {:.3f} | "
                  "Macro f1: {:.3f} | "
                  "Samples f1: {:.3f} | "
                  "Exact Match Accuracy: {:.3f} | ".format(val_loss,
                                                           result['micro_f1'],
                                                           result['micro_accuracy'],
                                                           result['micro_precision'],
                                                           result['micro_recall'],
                                                           result['macro_f1'],
                                                           result['samples_f1'],
                                                           result['exact_match_accuracy'])
                  )

            self.writer.add_scalars('Val Metrics', {'Loss': val_loss,
                                                     'Micro Accuracy': result['micro_accuracy'],
                                                     'Exact Match Accuracy': result['exact_match_accuracy'],
                                                     'Micro F1': result['micro_f1'],
                                                     'Macro F1': result['macro_f1']}, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                os.makedirs(os.path.dirname(self.args.checkpoint_dir), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state': self.state_dict(),
                    'model_optimizer': self.optimizer.state_dict(),
                    'model_scheduler': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, self.args.checkpoint_dir + '2d_conv_mfcc_classifier.pth')


    def testing_loop(self):
        test_set = IRMASDataset(self.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=self.args.n_mels,
                                n_mfcc=self.args.n_mfcc, name='test', sr=self.args.sr, return_type='mfcc',
                                use_window=True, window_size=3)

        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                                 collate_fn=du.collate_fn_windows)

        with torch.no_grad():
            outputs_list = []
            targets_list = []

            self.eval()
            for features, targets, lengths in tqdm(test_loader, desc='Testing', leave=False, total=len(test_loader)):

                current_audio_predictions = np.zeros_like(targets.cpu().numpy(), dtype=np.float32)
                # max_val = np.zeros(targets.shape[0], dtype=np.float32)

                # iterate over the first dimension of the features tensor
                for i in range(features.shape[0]):
                    data = features[i].to(self.device)
                    output = self.activation(self(data)).cpu().numpy()
                    # add output to current_audio_predictions at index i, but take into account only the first n lengths whenre n is given by i-th element of lengths tensor
                    current_audio_predictions[i] = np.sum(output[:lengths[i]], axis=0)

                # divide current_audio_predictions by maximum value of current_audio_predictions at the corresponding index of axis zero to normalize the values
                current_audio_predictions /= np.max(current_audio_predictions, axis=1, keepdims=True)
                # print("current_audio_predictions: ", current_audio_predictions)

                outputs_list.extend(current_audio_predictions)
                targets_list.extend(targets.cpu().numpy())

            test_loss = self.criterion(torch.tensor(outputs_list, dtype=torch.float32),
                                        torch.tensor(targets_list, dtype=torch.float32))
            result = calculate_metrics(np.array(outputs_list), np.array(targets_list))

            print("Testing metrics:")
            print("Loss: {:.3f} | "
                  "Micro f1: {:.3f} | "
                  "Micro Accuracy: {:.3f} | "
                  "Micro Precision: {:.3f} | "
                  "Micro Recall: {:.3f} | "
                  "Macro f1: {:.3f} | "
                  "Samples f1: {:.3f} | "
                  "Exact Match Accuracy: {:.3f} | ".format(test_loss,
                                                           result['micro_f1'],
                                                           result['micro_accuracy'],
                                                           result['micro_precision'],
                                                           result['micro_recall'],
                                                           result['macro_f1'],
                                                           result['samples_f1'],
                                                           result['exact_match_accuracy'])
                  )