import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
from data_utils import data_utils as du
from data_utils.IRMAS_dataloader import IRMASDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAccuracy
from tqdm import tqdm

NO_CLASSES = 11
THRESHOLD_VALUE = 0.5

DATA_MEAN = -0.000404580
DATA_STD = 0.108187131

tensorboard_loc = './tensorboard_results/1d_conv_raw_audio'

# create a model based on 1d convolutional neural network
class Conv1DModel(nn.Module):
    """
    Model based on 1d convolutional neural network
    
    Args:
        args: arguments from the command line
        
    """
    def __init__(self, args):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=440, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=6)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(4)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(4)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(128)
        self.pool6 = nn.MaxPool1d(4)
        self.conv7 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn7 = nn.BatchNorm1d(256)
        self.pool7 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(256, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, NO_CLASSES)
        self.relu = nn.ReLU()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)
        self.activation = nn.Sigmoid()
        self.writer = SummaryWriter(tensorboard_loc + f"_LR_{self.args.lr}_BATCH_SIZE_{self.args.batch_size}_SR_{self.args.sr}_TIME_{datetime.datetime.now()}")

        try:
            ckpt = torch.load(self.args.checkpoint_dir + '1d_conv_raw_audio_classifier.pth')
            self.start_epoch = ckpt['epoch'] + 1
            self.load_state_dict(ckpt['model_state'])
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
            self.conv1, self.relu, self.bn1, self.pool1, 
            self.conv2, self.relu, self.bn2, self.pool2, 
            self.conv3, self.relu, self.bn3, self.pool3, 
            self.conv4, self.relu, self.bn4, self.pool4, 
            self.conv5, self.relu, self.bn5, self.pool5, 
            self.conv6, self.relu, self.bn6, self.pool6, 
            self.conv7, self.relu, self.bn7, self.pool7
            )(x)


        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.squeeze(x, 2)
        x = nn.Sequential(self.fc1, self.dropout1, self.relu, self.fc2)(x)
        return x
    

# training loop
def train(model):
    # resample at 16kHz
    train_set = IRMASDataset(model.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='train', audio_augmentation=True, spectogram_augmentation=False, sr=model.args.sr, return_type='audio', audio_length=model.args.sr*3)
    train_loader = DataLoader(train_set, batch_size=model.args.batch_size, shuffle=True, drop_last=True)

    micro_accuracy = MultilabelAccuracy(NO_CLASSES, THRESHOLD_VALUE, average='micro').to(model.device)

    for epoch in range(model.start_epoch, model.args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for data, target in tqdm(train_loader, desc='Training', leave=False, total=len(train_loader)):
            data, target = data.to(model.device), target.to(model.device)
            model.optimizer.zero_grad()
            output = model(data)
            loss = model.criterion(model.activation(output), target)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
            # micro accuracy requires logit float outputs and applies sigmoid internally
            train_acc += micro_accuracy(output, target)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model.writer.add_scalars('Train metrics', {'loss': train_loss, 'micro_accuracy': train_acc}, epoch)
        print(f'\nEpoch: {epoch + 1}/{model.args.epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Micro Acc: {train_acc:.4f}')

        eval(model, epoch)

# evaluation loop
def eval(model, epoch):
    val_set = IRMASDataset(model.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='val', audio_augmentation=False, spectogram_augmentation=False, sr=model.args.sr, return_type='audio', use_window=True, window_size=3)
    
        
    val_loader = DataLoader(val_set, batch_size=model.args.batch_size, shuffle=False, drop_last=True, collate_fn=du.collate_fn_windows)

    with torch.no_grad():
        outputs_list = []
        targets_list = []

        model.eval()
        for features, targets, lengths in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

            current_audio_predictions = np.zeros_like(targets.cpu().numpy(), dtype=np.float32)
            #max_val = np.zeros(targets.shape[0], dtype=np.float32)

            # iterate over the first dimension of the features tensor
            for i in range(features.shape[0]):
                data = features[i].to(model.device)
                output = model.activation(model(data)).cpu().numpy()

                #print("\noutput: ", output[:lengths[i]])

                # set max_value at index i to max value across all dimensions of output
                #max_val[i] = np.max(output[:lengths[i]])

                #print("max_val: ", max_val)

                # add output to current_audio_predictions at index i, but take into account only the first n lengths whenre n is given by i-th element of lengths tensor
                current_audio_predictions[i] = np.sum(output[:lengths[i]], axis=0)

                #print("current_audio_predictions: ", current_audio_predictions[i])

                # divide current_audio_predictions by max_val to normalize the values
                #current_audio_predictions[i] /= max_val[i]

                #print("current_audio_predictions: ", current_audio_predictions[i])
            
            # divide current_audio_predictions by maximum value of current_audio_predictions at the corresponding index of axis zero to normalize the values
            current_audio_predictions /= np.max(current_audio_predictions, axis=1, keepdims=True)
            #print("current_audio_predictions: ", current_audio_predictions)

            outputs_list.extend(current_audio_predictions)
            targets_list.extend(targets.cpu().numpy())

    
        val_loss = model.criterion(torch.tensor(outputs_list, dtype=torch.float32), torch.tensor(targets_list, dtype=torch.float32))
        model.scheduler.step(val_loss)
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

        model.writer.add_scalars('Val Metrics', {'Loss': val_loss, 
                                                'Micro Accuracy': result['micro_accuracy'],
                                                'Exact Match Accuracy': result['exact_match_accuracy'],
                                                'Micro F1': result['micro_f1'],
                                                'Macro F1': result['macro_f1']}, epoch)
        
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            os.makedirs(os.path.dirname(model.args.checkpoint_dir), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'model_optimizer': model.optimizer.state_dict(),
                'model_scheduler': model.scheduler.state_dict(),
                'best_val_loss': model.best_val_loss
            }, model.args.checkpoint_dir + '1d_conv_raw_audio_classifier.pth')

def test(model):
    test_set = IRMASDataset(model.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='test', audio_augmentation=False, spectogram_augmentation=False, sr=model.args.sr, return_type='audio', use_window=True, window_size=3)
     
    test_loader = DataLoader(test_set, batch_size=model.args.batch_size, shuffle=False, drop_last=True, collate_fn=du.collate_fn_windows)

    with torch.no_grad():
        outputs_list = []
        targets_list = []

        model.eval()
        for features, targets, lengths in tqdm(test_loader, desc='Testing', leave=False, total=len(test_loader)):

            current_audio_predictions = np.zeros_like(targets.cpu().numpy(), dtype=np.float32)
            #max_val = np.zeros(targets.shape[0], dtype=np.float32)

            # iterate over the first dimension of the features tensor
            for i in range(features.shape[0]):
                data = features[i].to(model.device)
                output = model.activation(model(data)).cpu().numpy()
                # add output to current_audio_predictions at index i, but take into account only the first n lengths whenre n is given by i-th element of lengths tensor
                current_audio_predictions[i] = np.sum(output[:lengths[i]], axis=0)

            # divide current_audio_predictions by maximum value of current_audio_predictions at the corresponding index of axis zero to normalize the values
            current_audio_predictions /= np.max(current_audio_predictions, axis=1, keepdims=True)
            #print("current_audio_predictions: ", current_audio_predictions)

            outputs_list.extend(current_audio_predictions)
            targets_list.extend(targets.cpu().numpy())

    
        test_loss = model.criterion(torch.tensor(outputs_list, dtype=torch.float32), torch.tensor(targets_list, dtype=torch.float32))
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


def calculate_metrics(pred, target, threshold=0.5, no_classes=NO_CLASSES):
    pred = np.array(pred > threshold, dtype=int)
    micro_accuracy = MultilabelAccuracy(no_classes, threshold, average='micro')
    macro_accuracy = MultilabelAccuracy(no_classes, threshold, average='macro')


    return {
        'micro_accuracy': micro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'macro_accuracy': macro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'exact_match_accuracy': accuracy_score(target, pred),
        'micro_precision': precision_score(target, pred, average='micro', zero_division=0),
        'macro_precision': precision_score(target, pred, average='macro', zero_division=0),
        'macro_recall': recall_score(target, pred, average='macro', zero_division=0),
        'macro_f1': f1_score(target, pred, average='macro', zero_division=0),
        'micro_recall': recall_score(target, pred, average='micro', zero_division=0),
        'micro_f1': f1_score(target, pred, average='micro', zero_division=0),
        'samples_precision': precision_score(target, pred, average='samples', zero_division=0),
        'samples_recall': recall_score(target, pred, average='samples', zero_division=0),
        'samples_f1': f1_score(target, pred, average='samples', zero_division=0),
    }