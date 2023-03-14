import os
import ast

import pandas as pd
import numpy as np
import torch
import librosa as lr
import random
import scipy

from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from skimage.transform import resize

NO_CLASSES = 11


# convert
# dual-channel data to single channel
# this is not necessary if we're using librosa to load audio files, as it already converts all channels to mono!
def to_mono(x):
    if len(x.shape) == 1:
        return x
    elif (len(x.shape) == 2):
        return np.mean(x, axis=1)
    else:
        raise RuntimeError("Only one or two channel data is supported!")

def add_noise(y, sr, mean, std_dev, alpha=0.005):
    noise = np.random.normal(mean, std_dev, len(y))
    return y + alpha * noise

def pitch_shift(y, sr, range=6):
    return lr.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-range, range))


# shift limit is the maximum percentage of total audio length we want to be able to shift by
# i.e if it's 0.3, the audio file can maximally be shifted by 30% of its length
def time_shift(audio_file, shift_limit=0.4):
    sig_len = audio_file.shape[0]
    shift_amount = int(random.uniform(-1, 1) * shift_limit * sig_len)
    #print(shift_amount)
    return np.roll(audio_file, shift_amount)


def audio_to_spectogram(audio_file, sr, n_mels=128):
    spectogram = lr.feature.melspectrogram(y=audio_file, sr=sr, n_mels=n_mels)
    return spectogram

def spectogram_to_db(spectogram):
    return lr.power_to_db(spectogram, ref=np.max)

# adapted from https://www.kaggle.com/code/yash612/simple-audio-augmentation
def freq_mask(spec, F=None, num_masks=1):
    masked = spec.copy()
    num_mel_channels = masked.shape[0]

    # F denotes the maximum percentage of frequencies that will be masked
    if (F == None):
        F = int(masked.shape[0] / 10)

    for i in range(0, num_masks):
        freq = random.randrange(0, F)
        zero = random.randrange(0, num_mel_channels - freq)
        # avoids randrange error if values are equal and range is empty
        if (zero == zero + freq): return masked
        mask_end = random.randrange(zero, zero + freq)
        masked[zero:mask_end] = masked.mean()
    return masked


# adapted from https://www.kaggle.com/code/yash612/simple-audio-augmentation
def time_mask(spec, time=None, num_masks=1):
    masked = spec.copy()
    length = masked.shape[1]

    if (time == None):
        time = int(masked.shape[1] / 19)

    for i in range(0, num_masks):
        t = random.randrange(0, time)
        zero = random.randrange(0, length - t)
        if (zero == zero + t): return masked
        mask_end = random.randrange(zero, zero + t)
        masked[:, zero:mask_end] = masked.mean()
    return masked

def get_spectogram_transformation(height, width):
    transform_list = [
        Resize((height, width)),
        CenterCrop((height, width)),
        ToTensor()
    ]
    return Compose(transform_list)

class IRMASDataset(Dataset):
    def __init__(self, data_root_path, data_mean, data_std, n_mels=128, spec_height=None, name='train', audio_augmentation=False,
                 spectogram_augmentation=False, sr=44100, return_type="audio"):
        super(IRMASDataset, self).__init__()
        self.data_root_path = data_root_path
        self.audio_augmentation = audio_augmentation
        self.spectogram_augmentation = spectogram_augmentation
        self.name = name
        self.sr = sr
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_mels = n_mels
        self.spec_height = spec_height
        self.spec_width = self.n_mels
        self.return_type = return_type

        if self.spec_height == None:
            self.spec_height = self.n_mels

        assert name in ('train', 'test', 'val')
        assert return_type in ('audio', 'image')

        if name == 'train':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'train.csv'))
        elif name == 'test':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'test.csv'))
        elif name == 'val':
            self.examples = pd.read_csv(os.path.join(self.data_root_path, 'datalists', 'val.csv'))
        else:
            raise (f'{self.name} not defined')

    def __getitem__(self, index):
        audio_file_path = os.path.join(self.data_root_path, self.examples.iloc[[index]]["file_path"].item())
        audio_file, sr = lr.load(audio_file_path.encode('utf-8'), sr=self.sr)
        target_classes = self.examples.loc[[index]]["classes_id"].item()

        label_list = ast.literal_eval(target_classes)
        one_hot_vector = [0] * NO_CLASSES
        for i in label_list:
            one_hot_vector[i] = 1

        # TODO: provjeri je li ovaj target.float OK
        target = torch.tensor(one_hot_vector).float()

        assert sr == self.sr

        # normalize the audio file using mean and standard deviation computed over the training set
        audio_file = (audio_file - self.data_mean) / self.data_std

        if self.audio_augmentation:
            audio_file = add_noise(audio_file, sr, self.data_mean, self.data_std)
            audio_file = pitch_shift(audio_file, sr)
            audio_file = time_shift(audio_file)

        if self.return_type == 'audio':
            return torch.from_numpy(audio_file).float(), target

        spectogram = audio_to_spectogram(audio_file, self.sr, self.n_mels)

        # tenzor se prvo mora augmentirati, a tek onda skalirati!
        if self.spectogram_augmentation:
            spectogram = freq_mask(spectogram, int(spectogram.shape[0] / 10), 2)
            spectogram = time_mask(spectogram, int(spectogram.shape[1] / 10), 2)


        spectogram = spectogram_to_db(spectogram)
        #transform = get_spectogram_transformation(self.spec_height, self.spec_width)
        resized_spectogram = resize(spectogram, (self.spec_height, self.spec_width))

        spectogram_tensor = torch.from_numpy(resized_spectogram).float()
        spectogram_tensor = torch.unsqueeze(spectogram_tensor, 0)

        # ovaj repeat je samo da dobijemo rgb sliku iz 1-kanalnog spektograma
        # pa ponavljamo samo 1 kanal 3x
        return spectogram_tensor.repeat(3,1,1), target.float()

    def __len__(self):
        return len(self.examples)
