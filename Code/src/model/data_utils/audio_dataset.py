import ast
import math
import os

import librosa as lr
import numpy as np
import pandas as pd
import pyrootutils
import torch
from src.model.data_utils.data_utils import *
from torch.utils.data import Dataset

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class AudioDataset(Dataset):
    """
    Dataset class for all the datasets used.

    Args:
        data_root_path (str): path to the root folder of the dataset \\
        data_mean (float): mean of the training set audio files \\
        data_std (float): standard deviation of the training set audio files \\
        no_classes (int): number of classes in the dataset \\
        n_mels (int): number of mel bands to use for spectrogram and for resizing the mfccs \\
        n_mfcc (int): number of mfccs to use (only for `return_type` == "mfcc") \\
        spec_height (int): height of the spectrogram (only for `return_type` == "spectogram") \\
        name (str): name of the split to use (train, test, val) \\
        sr (int): sampling rate to use, not necessarily the original sampling rate of the files \\
        return_type (str): type of data to return (audio, spectogram, mfcc) \\
        use_window (bool): whether to use a window or not (suitable for both training and evaluation) \\
        window_size (int): size of the window (in seconds) to use (only for `use_window` == True) \\
        augmentation_config (dict): dictionary containing the augmentation configuration \\
        dynamic_sampling (bool): whether to use dynamic sampling or not \\
        min_sampled_files (int): minimum number of files to sample (only for `dynamic_sampling` == True) \\
        max_sampled_files (int): maximum number of files to sample (only for `dynamic_sampling` == True)
    """
    def __init__(self, data_root_path, data_mean, data_std, no_classes=11, n_mels=256, n_mfcc=40, spec_height=None, name='train', sr=44100, return_type="audio", use_window=False, window_size=None, augmentation_config=None, dynamic_sampling=False, min_sampled_files=None, max_sampled_files=None):
        super(AudioDataset, self).__init__()
        self.data_root_path = data_root_path
        self.name = name
        self.sr = sr
        self.data_mean = data_mean
        self.data_std = data_std
        self.no_classes = no_classes
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.spec_height = spec_height
        self.spec_width = self.n_mels
        self.return_type = return_type
        self.use_window = use_window
        self.window_size = window_size
        self.augmentation_config = augmentation_config
        self.dynamic_sampling = dynamic_sampling
        self.min_sampled_files = min_sampled_files
        self.max_sampled_files = max_sampled_files

        if self.spec_height is None:
            if self.return_type == "spectogram":
                self.spec_height = self.n_mels
            elif self.return_type == "mfcc":
                self.spec_height = self.n_mfcc

        assert name in ('train', 'test', 'val'), f'{name} not defined. Use train, test or val'
        assert return_type in ('audio', 'spectogram', 'mfcc', 'audio-features'), f'{return_type} not defined. Use audio, spectogram, mfcc or audio-features'

        if name == 'train':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'train.csv'))
        elif name == 'test':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'test.csv'))
        elif name == 'val':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'val.csv'))
        else:
            raise f'{self.name} not defined'

        if self.dynamic_sampling:
            self.label_to_indices = {}
            for i in range(len(self.examples)):
                target_classes = self.examples.loc[[i]]["classes_id"].item()
                label_list = ast.literal_eval(target_classes)
                for label in label_list:
                    if label not in self.label_to_indices:
                        self.label_to_indices[label] = []
                    self.label_to_indices[label].append(i)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the item to return
        Returns:
            if `use_window` == True:
                list: list containing audio files/spectrograms/mfccs split into windows
            else:
                torch.Tensor: tensor containing the audio file/spectrogram/mfcc
            """
        if self.dynamic_sampling:
            num_files = np.random.randint(self.min_sampled_files, self.max_sampled_files + 1)
            chosen_labels = np.random.choice(self.no_classes, num_files, replace=False)
            audio_file = None
            one_hot_vector = [0] * self.no_classes
            for label in chosen_labels:
                index = np.random.choice(self.label_to_indices[label])
                audio_file_path = os.path.join(self.data_root_path, self.examples.iloc[[index]]["file_path"].item())
                af, sr = lr.load(audio_file_path, sr=self.sr)
                audio_file = af if audio_file is None else (audio_file + af)
                target_classes = self.examples.loc[[index]]["classes_id"].item()
                label_list = ast.literal_eval(target_classes)
                for i in label_list:
                    one_hot_vector[i] = 1 if one_hot_vector[i] == 0 else one_hot_vector[i]
            audio_file /= num_files
        else:
            audio_file_path = os.path.join(self.data_root_path, self.examples.iloc[[index]]["file_path"].item())
            audio_file, sr = lr.load(audio_file_path, sr=self.sr)
            target_classes = self.examples.loc[[index]]["classes_id"].item()

            label_list = ast.literal_eval(target_classes)
            one_hot_vector = [0] * self.no_classes
            for i in label_list:
                one_hot_vector[i] = 1

        target = torch.tensor(one_hot_vector).float()

        assert sr == self.sr

        # normalize the audio file using mean and standard deviation computed over the training set
        audio_file = (audio_file - self.data_mean) / self.data_std

        if self.augmentation_config.audio.active:
            audio_file = augment_audio(audio_file, sr, self.augmentation_config)

        audio_windows = []
        num_intervals = 0

        if self.use_window:
            samples_per_interval = self.sr * self.window_size
            num_intervals = int(np.ceil(len(audio_file) / samples_per_interval))

            for i in range(num_intervals):
                start = i * samples_per_interval
                end = min(start + samples_per_interval, len(audio_file))
                interval_audio = audio_file[start:end]
                # pad the last window with zeros as it's most likely going to be shorter than other windows
                if i == (num_intervals - 1):
                    interval_audio = np.pad(interval_audio, (0, samples_per_interval - len(interval_audio)), "constant")
                audio_windows.append(interval_audio)

        if self.return_type == 'audio':
            if self.use_window:
                # return a list of audio windows as float tensor
                return [torch.from_numpy(audio_window).float().view(1, -1) for audio_window in
                        audio_windows], target, num_intervals
            else:
                return torch.from_numpy(audio_file).float().view(1, -1), target

        if self.return_type == 'audio-features':
            if self.use_window:
                return [get_audio_features(audio, sr=self.sr) for audio in audio_windows], target, num_intervals
            else:
                return get_audio_features(audio_file, sr=self.sr), target

        elif self.return_type == 'mfcc':
            if self.use_window:
                return [get_mfcc(audio, sr=self.sr, n_mfcc=self.n_mfcc, mfcc_height=self.spec_height,
                                 mfcc_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active,
                                 config=self.augmentation_config) for audio in audio_windows], target, num_intervals
            else:
                return get_mfcc(audio_file, sr=self.sr, n_mfcc=self.n_mfcc, mfcc_height=self.spec_height,
                                mfcc_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active,
                                config=self.augmentation_config), target

        if self.use_window:
            return [get_spectogram(audio, sr=self.sr, n_mels=self.n_mels, spec_height=self.spec_height,
                                   spec_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active,
                                   config=self.augmentation_config) for audio in audio_windows], target, num_intervals
        else:
            return get_spectogram(audio_file, sr=self.sr, n_mels=self.n_mels, spec_height=self.spec_height,
                                  spec_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active,
                                  config=self.augmentation_config), target

    def __len__(self):
        """
        Returns:
            int: number of items in the dataset
        """
        if not self.dynamic_sampling:
            return len(self.examples)
        else:
            n = len(self.examples)
            # this formula ensures that only 0.1% of the files will not be sampled and the estimation is better as n increases
            # for n = 6000, the formula returns ~ 2.3*6000
            return int(math.log(0.001, 1 - 1 / n) * 1 / n * 2 / (self.min_sampled_files + self.max_sampled_files) * n)