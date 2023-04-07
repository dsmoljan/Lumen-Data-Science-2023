import librosa as lr
import numpy as np
import torch

import random

from skimage.transform import resize


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
    # print(shift_amount)
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

def get_spectogram(audio_file, sr, n_mels, spec_height, spec_width, augmentation=False):
    spectogram = audio_to_spectogram(audio_file, sr, n_mels)

    # tenzor se prvo mora augmentirati, a tek onda skalirati!
    if augmentation:
        spectogram = freq_mask(spectogram, int(spectogram.shape[0] / 10), 2)
        spectogram = time_mask(spectogram, int(spectogram.shape[1] / 10), 2)

    spectogram = spectogram_to_db(spectogram)
    resized_spectogram = resize(spectogram, (spec_height, spec_width))

    spectogram_tensor = torch.from_numpy(resized_spectogram).float()
    spectogram_tensor = torch.unsqueeze(spectogram_tensor, 0)

    # ovaj repeat je samo da dobijemo rgb sliku iz 1-kanalnog spektograma
    # pa ponavljamo samo 1 kanal 3x
    return spectogram_tensor.repeat(3, 1, 1)

def get_mfcc(audio_file, sr, n_mfcc, spec_height, spec_width, augmentation=False):
    mfcc = lr.feature.mfcc(y=audio_file, sr=sr, n_mfcc=n_mfcc)

    if augmentation:
        mfcc = freq_mask(mfcc, int(mfcc.shape[0] / 10), 2)
        mfcc = time_mask(mfcc, int(mfcc.shape[1] / 10), 2)

    # TODO do we want to resize mfcc? initial width is 259 (i don't know why), resized it is 256 (default n_mels)
    resized_mfcc = resize(mfcc, (spec_height, spec_width))

    mfcc_tensor = torch.from_numpy(resized_mfcc).float()
    mfcc_tensor = torch.unsqueeze(mfcc_tensor, 0)

    # return mfcc_tensor.repeat(3,1,1)
    # why would we make it 3 channels if it is effectively 1 channel?
    return mfcc_tensor

def collate_fn_windows(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    examples, labels, lengths = zip(*data)
    max_len = max(lengths)
    example_shape = examples[0][0].size()
    # features shape should be [batch_size, max_window_num, 3, 128, 128] for spectograms
    # and for audio, their shape should be [batch_size, max_window_num, 1, SR*window_size]
    # and for MFCC, their shape should be [batch_size, max_window_num, 3, 13, 128]
    features = torch.zeros((len(data), max_len, *example_shape))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j = lengths[i]
        emtpy_tensor = torch.zeros((max_len - j, *example_shape))
        features[i] = torch.cat((torch.stack(examples[i]), emtpy_tensor))

    return features.float(), labels.long(), lengths.long()