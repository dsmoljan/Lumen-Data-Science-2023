import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from data_utils.IRMAS_dataloader import IRMASDataset
import os
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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=9)
        self.relu = nn.ReLU()
        self.pool16 = nn.MaxPool1d(kernel_size=16)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3= nn.Linear(64, NO_CLASSES)
        self.dropout2 = nn.Dropout(0.2)
        self.args = args
        self.writer = SummaryWriter(tensorboard_loc + '_classifier')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, verbose=True)

        try:
            ckpt = torch.load(self.args.checkpoint_dir + '1d_conv_raw_audio_classifier.pth')
            self.start_epoch = ckpt['epoch']
            self.load_state_dict(ckpt['model_state'])
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.optimizer.load_state_dict(ckpt['model_optimizer'])
            self.best_val_loss = ckpt['best_val_loss']
        except Exception as e:
            print(' [*] No checkpoint!')
            self.start_epoch = 0
            self.best_val_loss = np.inf

        self.to(self.device)

    def forward(self, x):
        num_samples = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool16(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout1(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout1(x)
        x = x.view(num_samples, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        x = (x > THRESHOLD_VALUE).float()
        return x
    
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
    
    def predict_logits(self, x):
        return self.forward(x)
    

# training loop
def train(model):
    # resample at 16kHz
    train_set = IRMASDataset(model.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='train', audio_augmentation=True, spectogram_augmentation=False, sr=16000, return_type='audio', audio_length=16000*3)
    train_loader = DataLoader(train_set, batch_size=model.args.batch_size, shuffle=True, drop_last=True)

    micro_accuracy = MultilabelAccuracy(NO_CLASSES, THRESHOLD_VALUE, average='micro')

    for epoch in range(model.start_epoch, model.args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for data, target in tqdm(train_loader, desc='Training', leave=False, total=len(train_loader)):
            data, target = data.to(model.device), target.to(model.device)
            model.optimizer.zero_grad()
            output = model(data)
            loss = model.criterion(torch.sigmoid(output), target)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
            train_acc += micro_accuracy(output, target)
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        model.writer.add_scalars('Train metrics', {'loss': train_loss, 'micro_accuracy': train_acc}, epoch)
        print(f'\nEpoch: {epoch + 1}/{model.args.epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Micro Acc: {train_acc:.4f}')

        eval(model, epoch)

# evaluation loop
def eval(model, epoch):
    val_set = IRMASDataset(model.args.data_root_path, DATA_MEAN, DATA_STD, n_mels=256, name='val', audio_augmentation=False, spectogram_augmentation=False, sr=16000, return_type='audio', audio_length=16000*3)
    val_loader = DataLoader(val_set, batch_size=model.args.batch_size, shuffle=True, drop_last=True)
    val_loss = 0

    with torch.no_grad():
        outputs_list = []
        targets_list = []

        model.eval()
        for data, target in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):
            data, target = data.to(model.device), target.to(model.device)
            output = torch.sigmoid(model(data))
            outputs_list.extend(output.cpu().numpy())
            targets_list.extend(target.cpu().numpy())
            loss = model.criterion(output, target)
            val_loss += loss.item()

    
        val_loss /= len(val_loader)
        model.scheduler.step(val_loss)
        result = calculate_metrics(np.array(outputs_list), np.array(targets_list))

        print("Validation metrics:")
        print("Loss: {:.3f} | "
              "Micro f1: {:.3f} | "
              "Macro f1: {:.3f} | "
              "Micro Accuracy: {:.3f} | "
              "Exact Match Accuracy: {:.3f} | "
              "Micro Precision: {:.3f} | "
              "Micro Recall: {:.3f} | ".format(val_loss,
                                               result['micro_f1'],
                                               result['macro_f1'],
                                               result['micro_accuracy'],
                                               result['exact_match_accuracy'],
                                               result['micro_precision'],
                                               result['micro_recall'])
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
                'best_val_loss': model.best_val_loss
            }, model.args.checkpoint_dir + '1d_conv_raw_audio_classifier.pth')


def calculate_metrics(pred, target, threshold=0.5, no_classes=NO_CLASSES):
    pred = np.array(pred > threshold, dtype=int)
    micro_accuracy = MultilabelAccuracy(no_classes, threshold, average='micro')
    macro_accuracy = MultilabelAccuracy(no_classes, threshold, average='macro')


    return {
        'micro_accuracy': micro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'macro_accuracy': macro_accuracy(torch.from_numpy(pred), torch.from_numpy(target)),
        'micro_precision': precision_score(target, pred, average='micro'),
        'macro_precision': precision_score(target, pred, average='macro'),
        'micro_recall': recall_score(target, pred, average='micro'),
        'macro_recall': recall_score(target, pred, average='macro'),
        'micro_f1': f1_score(target, pred, average='micro'),
        'macro_f1': f1_score(target, pred, average='macro'),
        'exact_match_accuracy': accuracy_score(target, pred)
    }