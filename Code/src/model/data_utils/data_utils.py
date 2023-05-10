import random

import librosa as lr
import numpy
import numpy as np
import torch
from skimage.transform import resize


def to_mono(x: numpy.ndarray) -> numpy.ndarray:
    """
    Converts dual-channel audio signal to single channel. Not necessary if Librosa is used to load audio files
    as it already converts all channels to mono. Maximum number of supported original channels is 2

    Args:
        x (numpy array): audio signal to convert to mono
    
    Returns:
        numpy array: mono audio signal
    
    Raises:
        RuntimeError: if the given signal has more than 2 channels
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
        y (numpy array): audio signal to add noise to \\
        mean (float): mean value of the Gaussian distribution used to generate noise \\
        std_dev (float): standard deviation of the Gaussian distribution used to generate noise \\
        alpha (float): factor by which the noise signal is multiplied with; the smaller alpha is, the weaker the noise is
               and stronger the original signal

    Returns:
        numpy array: the original signal with added noise
    """
    noise = np.random.normal(mean, std_dev, len(y))
    return y + alpha * noise


def pitch_shift(y: np.ndarray, sr: int, range=6) -> numpy.ndarray:
    """
    Shifts the given signal's frequency up or down

    Args:
        y (numpy array): audio signal to shift \\
        sr (int): sampling rate of the audio signal \\
        range (int): number of semitones we want to shift the signal by. If positive, the signal is shifted up. If negative, the signal is shifted down \\
    
    Returns:
        numpy array: the shifted signal
    """
    return lr.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-range, range))


def time_shift(audio_file: numpy.ndarray, shift_limit=0.4) -> numpy.ndarray:
    """
    Randomly shifts the given signal in its time domain, upper bound by a given percentage. I.e if the shift limit is 0.3,
    the last 30% of the audio signal now become the first 30%

    Args:
        audio_file (numpy array): the signal to be shifted in the form of a Numpy array \\
        shift_limit (float): the maximum percentage of total audio length to be shifted by. The actual limit is chosen as random.uniform(-shift_limit, shift_limit)
    
    Returns:
        numpy array: the shifted signal
    """
    sig_len = audio_file.shape[0]
    shift_amount = int(random.uniform(-1, 1) * shift_limit * sig_len)
    # print(shift_amount)
    return np.roll(audio_file, shift_amount)


def audio_to_spectogram(audio_file: numpy.ndarray, sr: int, n_mels=128) -> numpy.ndarray:
    """
    Converts an audio file in the form of a numpy array to a spectogram.

    Args:
        audio_file (numpy array): audio file in the form of a Numpy array \\
        sr (int): sampling rate of the audio file \\
        n_mels (int): number of Mel bands to generate

    Returns:
        numpy array: spectogram
    """
    spectogram = lr.feature.melspectrogram(y=audio_file, sr=sr, n_mels=n_mels)
    return spectogram


def spectogram_to_db(spectogram: numpy.ndarray) -> numpy.ndarray:
    """
    Wrapper around the lr.power_to_db method.

    Args:
        spectogram (numpy array): spectogram to convert to decibel scale

    Returns:
        numpy array: spectogram in decibel scale
    """
    return lr.power_to_db(spectogram, ref=np.max)


def freq_mask(spec: numpy.ndarray, F_per=10, num_masks=1) -> numpy.ndarray:
    """
    Implementation of the spectogram frequency masking augmentation technique, introduced in
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html. Random parts of the spectogram
    frequency domain are selected and replaced with the mean value from the entire spectogram.

    Implementation adapted from: https://www.kaggle.com/code/yash612/simple-audio-augmentation

    Args:
        spec (numpy array): spectogram to frequency maks \\
        F_per (int): maximum percentage of frequencies that will be masked. Default = 10 percent \\
        num_masks (int): number of frequency masks to use.

    Returns:
        numpy array: spectogram with frequency masks applied
    """
    masked = spec.copy()
    num_mel_channels = masked.shape[0]

    # F denotes the maximum percentage of frequencies that will be masked
    F = int(masked.shape[0] / F_per)

    for i in range(0, num_masks):
        freq = random.randrange(0, F)
        zero = random.randrange(0, num_mel_channels - freq)
        # avoids randrange error if values are equal and range is empty
        if (zero == zero + freq): return masked
        mask_end = random.randrange(zero, zero + freq)
        masked[zero:mask_end] = masked.mean()
    return masked


# adapted from https://www.kaggle.com/code/yash612/simple-audio-augmentation
def time_mask(spec: numpy.ndarray, time_per=10, num_masks=1) -> numpy.ndarray:
    """
    Implementation of the spectogram time masking augmentation technique, introduced in
    https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html. Random parts of the spectogram
    time domain are selected and replaced with the mean value from the entire spectogram.

    Implementation adapted from: https://www.kaggle.com/code/yash612/simple-audio-augmentation

    Args:
        spec (numpy array): spectogram to time mask \\
        time_per (int): maximum percentage of time domain that will be masked. Default = 10 percent \\
        num_masks (int): number of time masks to use.

    Returns:
        numpy array: spectogram with time masks applied
    """
    masked = spec.copy()
    length = masked.shape[1]

    time = int(masked.shape[1] / time_per)

    for i in range(0, num_masks):
        t = random.randrange(0, time)
        zero = random.randrange(0, length - t)
        if (zero == zero + t): return masked
        mask_end = random.randrange(zero, zero + t)
        masked[:, zero:mask_end] = masked.mean()
    return masked


def augment_audio(audio_file: numpy.ndarray, sr=None, config=None)->numpy.ndarray:
    """
    Helper method to organize audio augmentation logic and enable easier Hydra integration.

    Args:
        audio_file (numpy array): audio file in the form of a Numpy array \\
        sr (int): sampling rate of the audio file \\
        config (dict): configuration dictionary

    Returns:
        numpy array: augmented audio file
    """
    if config.audio.add_noise.active:
        audio_file = add_noise(audio_file, config.audio.add_noise.mean, config.audio.add_noise.std, alpha=config.audio.add_noise.alpha)
    if config.audio.pitch_shift.active:
        audio_file = pitch_shift(audio_file, sr)
    if config.audio.time_shift.active:
        audio_file = time_shift(audio_file, shift_limit=config.audio.time_shift.shift_limit)
    return audio_file

def get_spectogram(audio_file: numpy.ndarray, sr: int, n_mels: int, spec_height: int, spec_width: int, augmentation=False, config=None) -> torch.Tensor:
    """
    Constructs a spectogram tensor using the given audio signal and, optionally, augments it.

    Args:
        audio_file (numpy array): audio signal used to create the spectogram \\
        sr (int): sampling rate of the audio signal \\
        n_mels (int): number of Mel bands to generate \\
        spec_height (int): height to which to resize the spectogram \\
        spec_width (int): width to which to resize the spectogram \\
        augmentation (bool): indicates whether to use frequency and time masking augmentation on the spectogram \\
        
    Returns:
        torch Tensor: spectogram tensor
    """
    spectogram = audio_to_spectogram(audio_file, sr, n_mels)

    # tenzor se prvo mora augmentirati, a tek onda skalirati!
    if augmentation:
        if config.spectogram.freq_mask.active:
            spectogram = freq_mask(spectogram, config.spectogram.freq_mask.F_per, config.spectogram.freq_mask.num_masks)
        if config.spectogram.time_mask.active:
            spectogram = time_mask(spectogram, config.spectogram.time_mask.time_per, config.spectogram.time_mask.num_masks)

    spectogram = spectogram_to_db(spectogram)
    resized_spectogram = resize(spectogram, (spec_height, spec_width))

    spectogram_tensor = torch.from_numpy(resized_spectogram).float()
    spectogram_tensor = torch.unsqueeze(spectogram_tensor, 0)

    # ovaj repeat je samo da dobijemo rgb sliku iz 1-kanalnog spektograma
    # pa ponavljamo samo 1 kanal 3x
    return spectogram_tensor.repeat(3, 1, 1)


def get_mfcc(audio_file: numpy.ndarray, sr: int, n_mfcc: int, mfcc_height: int, mfcc_width: int, augmentation=False, config=None) -> torch.Tensor:
    """
    Generates Mel-frequency cepstral coefficients tensor of the given audio signal.

    Args:
        audio_file (numpy array): audio signal used to calculate MFCC \\
        sr (int): sampling rate of the audio signal \\
        n_mfcc (int): number of MFCC's to return \\
        mfcc_height (int): height to which to resize the computed MFCC \\
        mfcc_width (int): width to which to resize the computed MFCC \\
        augmentation (bool): indicates whether to use frequency and time masking augmentation on the MFCC \\
        
    Returns:
        torch Tensor: MFCC tensor
    """
    mfcc = lr.feature.mfcc(y=audio_file, sr=sr, n_mfcc=n_mfcc)

    if augmentation:
        mfcc = freq_mask(mfcc, config.spectogram.freq_mask.F_per, config.spectogram.freq_mask.num_masks)
        mfcc = time_mask(mfcc, config.spectogram.time_mask.time_per, config.spectogram.time_mask.num_masks)

    # TODO do we want to resize mfcc? initial width is 259 (i don't know why), resized it is 256 (default n_mels)
    resized_mfcc = resize(mfcc, (mfcc_height, mfcc_width))

    mfcc_tensor = torch.from_numpy(resized_mfcc).float()
    mfcc_tensor = torch.unsqueeze(mfcc_tensor, 0)

    # return mfcc_tensor.repeat(3,1,1)
    # why would we make it 3 channels if it is effectively 1 channel?
    return mfcc_tensor


def get_audio_features(y: numpy.ndarray, sr: int) -> torch.Tensor:
    mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = lr.feature.chroma_stft(y=y, sr=sr)
    zcr = lr.feature.zero_crossing_rate(y=y)
    spec_cent = lr.feature.spectral_centroid(y=y, sr=sr)
    rms = lr.feature.rms(y=y)
    spec_contrast = lr.feature.spectral_contrast(y=y, sr=sr)
    spec_bandwidth = lr.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = lr.feature.spectral_rolloff(y=y, sr=sr)
    spec_flatness = lr.feature.spectral_flatness(y=y)
    poly_features = lr.feature.poly_features(y=y, sr=sr)

    features = np.vstack((mfcc, chroma, zcr, spec_cent, rms, spec_contrast, spec_bandwidth, spec_rolloff, spec_flatness, poly_features))
    return torch.from_numpy(features).float()


def collate_fn_windows(data):
    """
    Collate function for DataLoader that pads the examples in a batch to the same length

    Args:
        data (list): list of tuples (example, label, length) where example is a tensor of arbitrary shape and label/length are scalars

    Returns:
        tuple: tuple of padded tensors (examples, labels, lengths) where examples is a tensor of shape `[batch_size, max_window_num, 3, 128, 128]` for spectograms, `[batch_size, max_window_num, 1, SR*window_size]` for audio, `[batch_size, max_window_num, 3, 40, 256]` for MFCC
    """
    examples, labels, lengths = zip(*data)
    max_len = max(lengths)
    example_shape = examples[0][0].size()
    # features shape should be [batch_size, max_window_num, 3, 128, 128] for spectograms
    # and for audio, their shape should be [batch_size, max_window_num, 1, SR*window_size]
    # and for MFCC, their shape should be [batch_size, max_window_num, 3, 40, 256]
    features = torch.zeros((len(data), max_len, *example_shape))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j = lengths[i]
        emtpy_tensor = torch.zeros((max_len - j, *example_shape))
        features[i] = torch.cat((torch.stack(examples[i]), emtpy_tensor))

    return features.float(), labels.long(), lengths.long()


def collate_fn_windows_stack(data):
    """
    A collate function which simply stacks all tensors. It assumes all examples
    are of the exact same length!

    Args:
        data (list): list of tuples (example, label, length) where example is a tensor of arbitrary shape and label/length are scalars

    Returns:
        tuple: tuple of stacked tensors
    """
    examples, labels, lengths = zip(*data)
    examples_tensor_list = [torch.stack(t_list) for t_list in examples]
    result_tensor = torch.cat(examples_tensor_list, dim=0)
    label_tuple = torch.stack(labels)
    num_windows = len(examples[0])
    label_list = [label for label in label_tuple for _ in range(num_windows)]
    labels_tensor = torch.stack(label_list, dim=0)

    return result_tensor.float(), labels_tensor.float()
