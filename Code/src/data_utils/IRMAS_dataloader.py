import ast
import math
import os

import pandas as pd
import pyrootutils
from src.data_utils.data_utils import *
from torch.utils.data import Dataset

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

NO_CLASSES = 11

class IRMASDataset(Dataset):
    def __init__(self, data_root_path, data_mean, data_std, n_mels=128, n_mfcc=13, spec_height=None, name='train',
                 mfcc_augmentation=False, sr=44100, return_type="audio",
                 use_window=False, window_size=None, augmentation_config=None, dynamic_sampling=False, min_sampled_files=None, max_sampled_files=None):
        super(IRMASDataset, self).__init__()
        self.data_root_path = data_root_path
        self.mfcc_augmentation = mfcc_augmentation
        self.name = name
        self.sr = sr
        self.data_mean = data_mean
        self.data_std = data_std
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

        assert name in ('train', 'test', 'val')
        assert return_type in ('audio', 'spectogram', 'mfcc')

        if name == 'train':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'train.csv'))
        elif name == 'test':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'test.csv'))
        elif name == 'val':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'val.csv'))
        else:
            raise f'{self.name} not defined'

    def __getitem__(self, index):
        if self.dynamic_sampling:
            num_files = np.random.randint(self.min_sampled_files, self.max_sampled_files + 1)
            indices = np.random.choice(len(self.examples), num_files, replace=True)
            audio_file = None
            one_hot_vector = [0] * NO_CLASSES
            for index in indices:
                audio_file_path = os.path.join(self.data_root_path, self.examples.iloc[[index]]["file_path"].item())
                af, sr = lr.load(audio_file_path, sr=self.sr)
                # TODO: ima li bolji nacin da se spoje audio fajlovi osim addition? jel okej pristup da se prvo spoje audio fajlovi pa se onda radi normalizacija, augmentacija, spektogrami/mfcc
                audio_file = af if audio_file is None else (audio_file + af)
                target_classes = self.examples.loc[[index]]["classes_id"].item()
                label_list = ast.literal_eval(target_classes)
                for i in label_list:
                    one_hot_vector[i] = 1 if one_hot_vector[i] == 0 else one_hot_vector[i]
        else:   
            audio_file_path = os.path.join(self.data_root_path, self.examples.iloc[[index]]["file_path"].item())
            audio_file, sr = lr.load(audio_file_path, sr=self.sr)
            target_classes = self.examples.loc[[index]]["classes_id"].item()

            label_list = ast.literal_eval(target_classes)
            one_hot_vector = [0] * NO_CLASSES
            for i in label_list:
                one_hot_vector[i] = 1

        # TODO: trenutno je moguce da se od n sampleanih neki preklapaju pa od npr. 5 samplanih budu 4 pozitivne labele, zelimo li ovo promijeniti?
        target = torch.tensor(one_hot_vector).float()

        assert sr == self.sr

        # normalize the audio file using mean and standard deviation computed over the training set
        # TODO do we apply the same normalization when overlapping the files in dynamic sampling?
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
                    interval_audio = np.pad(audio_file, (0, samples_per_interval - len(interval_audio)), "constant")
                audio_windows.append(interval_audio)

        if self.return_type == 'audio':
            if self.use_window:
                # return a list of audio windows as float tensor
                return [torch.from_numpy(audio_window).float().view(1, -1) for audio_window in
                        audio_windows], target, num_intervals
            else:
                return torch.from_numpy(audio_file).float().view(1, -1), target

        elif self.return_type == 'mfcc':
            if self.use_window:
                return [get_mfcc(audio, sr=self.sr, n_mfcc=self.n_mfcc, mfcc_height=self.spec_height, mfcc_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active, config=self.augmentation_config) for audio in audio_windows], target, num_intervals
            else:
                return get_mfcc(audio_file, sr=self.sr, n_mfcc=self.n_mfcc, mfcc_height=self.spec_height, mfcc_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active, config=self.augmentation_config), target

        if self.use_window:
            return [get_spectogram(audio, sr=self.sr, n_mels=self.n_mels, spec_height=self.spec_height, spec_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active, config=self.augmentation_config) for audio in audio_windows], target, num_intervals
        else:
            return get_spectogram(audio_file, sr=self.sr, n_mels=self.n_mels, spec_height=self.spec_height, spec_width=self.spec_width, augmentation=self.augmentation_config.spectogram.active, config=self.augmentation_config), target

    def __len__(self):
        if not self.dynamic_sampling:
            return len(self.examples)
        else:
            n = len(self.examples)
            # this formula ensures that only 0.1% of the files will not be sampled and the estimation is better as n increases
            # for n = 6000, the forula returns ~ 2.3*6000
            return int(math.log(0.001, 1 - 1/n) * 1/n * 2/(self.min_sampled_files + self.max_sampled_files) * n)
