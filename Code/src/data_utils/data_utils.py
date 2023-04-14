import librosa as lr
import numpy
import numpy as np
import torch

import random

from skimage.transform import resize


def to_mono(x: numpy.ndarray) -> numpy.ndarray:
    """
    Converts dual-channel audio signal to single channel. Not necessary if Librosa is used to load audio files
    as it already converts all channels to mono. Maximum number of supported original channels is 2

    :param x: numpy array - audio signal to convert to mono
    :return: the original signal converted to mono channel
    :raise: RuntimeError - if the input signal is comprised of more than 2 audio channels
    """
    if len(x.shape) == 1:
        return x
    elif (len(x.shape) == 2):
        return np.mean(x, axis=1)
    else:
        raise RuntimeError("Only one or two channel data is supported!")


def add_noise(y: np.ndarray, mean: float, std_dev: float, alpha=0.005) -> numpy.ndarray:
    """
    Adds Gaussian noise to a given signal y.

    Args:
        y: numpy array - audio signal to add noise to
        mean: mean value of the Gaussian distribution used to generate noise
        std_dev: standard deviation of the Gaussian distribution used to generate noise
        alpha: factor by which the noise signal is multiplied with; the smaller alpha is, the weaker the noise is
               and stronger the original signal
    """
    noise = np.random.normal(mean, std_dev, len(y))
    return y + alpha * noise


def pitch_shift(y: np.ndarray, sr: int, range=6) -> numpy.ndarray:
    """
    Shifts the given signal's frequency up or down

    :param y: numpy array representing an audio signal whose pitch we want to shift
    :param sr: sampling rate of the signal
    :param range: number of semitones we want to shift the signal by. If positive, the signal is shifted up.
                  If negative, the signal is shifted down
    :return: the shifted signal
    """
    return lr.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-range, range))


def time_shift(audio_file: numpy.ndarray, shift_limit=0.4) -> numpy.ndarray:
    """
    Randomly shifts the given signal in its time domain, upper bound by a given percentage. I.e if the shift limit is 0.3,
    the last 30% of the audio signal now become the first 30%

    :param audio_file: the signal to be shifted in the form of a Numpy array
    :param shift_limit: the maximum percentage of total audio length to be shifted by. The actual limit is chosen as
                        random.uniform(-shift_limit, shift_limit)
    :return: the shifted signal
    """
    sig_len = audio_file.shape[0]
    shift_amount = int(random.uniform(-1, 1) * shift_limit * sig_len)
    # print(shift_amount)
    return np.roll(audio_file, shift_amount)


def audio_to_spectogram(audio_file: numpy.ndarray, sr: int, n_mels=128) -> numpy.ndarray:
    """
    Converts an audio file in the form of a numpy array to a spectogram.

    :param audio_file: audio file in the form of a Numpy array
    :param sr: sampling rate of the audio file
    :param n_mels: number of Mel bands to generate
    :return: spectogram
    """
    spectogram = lr.feature.melspectrogram(y=audio_file, sr=sr, n_mels=n_mels)
    return spectogram


def spectogram_to_db(spectogram: numpy.ndarray) -> numpy.ndarray:
    """
    Wrapper around the lr.power_to_db method.
    """
    return lr.power_to_db(spectogram, ref=np.max)


def freq_mask(spec: numpy.ndarray, F=None, num_masks=1) -> numpy.ndarray:
    """
    Implementation of the spectogram frequency masking augmentation technique, introduced in
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html. Random parts of the spectogram
    frequency domain are selected and replaced with the mean value from the entire spectogram.

    Implementation adapted from: https://www.kaggle.com/code/yash612/simple-audio-augmentation

    :param spec: spectogram to frequency maks
    :param F: maximum percentage of frequencies that will be masked. Default = 0.1
    :param num_masks: number of frequency masks to use
    :return:
    """
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
def time_mask(spec: numpy.ndarray, time=None, num_masks=1) -> numpy.ndarray:
    """
    Implementation of the spectogram time masking augmentation technique, introduced in
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html. Random parts of the spectogram
    time domain are selected and replaced with the mean value from the entire spectogram.

    Implementation adapted from: https://www.kaggle.com/code/yash612/simple-audio-augmentation

    :param spec: spectogram to time mask
    :param time: maximum percentage of time domain that will be masked. Default = 0.1
    :param num_masks: number of time masks to use
    :return:
    """
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


def get_spectogram(audio_file: numpy.ndarray, sr: int, n_mels: int, spec_height: int, spec_width: int, augmentation=False) -> numpy.ndarray:
    """
    Constructs a spectogram using the given audio signal and, optionally, augments it.

    :param audio_file: audio signal used to create the spectogram
    :param sr: sampling rate of the audio signal
    :param n_mels: number of Mel bands to generate
    :param spec_height: height to which to resize the spectogram
    :param spec_width: width to which to resize the spectogram
    :param augmentation: indicates whether to use frequency and time masking augmentation on the spectogram
    :return: spectogram
    """
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


def get_mfcc(audio_file: numpy.ndarray, sr: int, n_mfcc: int, mfcc_height: int, mfcc_width: int, augmentation=False) -> torch.Tensor:
    """
    Generates Mel-frequency cepstral coefficients of the given audio signal.

    :param audio_file: audio signal used to calculaten MFCC
    :param sr: sampling rate of the audio signal
    :param n_mfcc: number of MFCC's to return
    :param mfcc_height: height to which to resize the computed MFCC
    :param mfcc_width: width to which to resize the computed MFCC
    :param augmentation: indictaes whether to use frequency and time masking augmentation on the MFCC
    :return:
    """
    mfcc = lr.feature.mfcc(y=audio_file, sr=sr, n_mfcc=n_mfcc)

    if augmentation:
        mfcc = freq_mask(mfcc, int(mfcc.shape[0] / 10), 2)
        mfcc = time_mask(mfcc, int(mfcc.shape[1] / 10), 2)

    # TODO do we want to resize mfcc? initial width is 259 (i don't know why), resized it is 256 (default n_mels)
    resized_mfcc = resize(mfcc, (mfcc_height, mfcc_width))

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
