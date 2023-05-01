import numpy as np
import librosa as lr
import torch
from skimage.transform import resize


def split_into_windows(audio_file: np.ndarray, sr: int, window_size: int):
    audio_windows = []
    samples_per_interval = sr * window_size
    num_intervals = int(np.ceil(len(audio_file) / samples_per_interval))

    for i in range(num_intervals):
        start = i * samples_per_interval
        end = min(start + samples_per_interval, len(audio_file))
        interval_audio = audio_file[start:end]
        # pad the last window with zeros as it's most likely going to be shorter than other windows
        if i == (num_intervals - 1):
            interval_audio = np.pad(interval_audio, (0, samples_per_interval - len(interval_audio)), "constant")
        audio_windows.append(interval_audio)

    return audio_windows

def preprocess_spectogram(audio_file: np.ndarray, sr: int, n_mels: int, height: int, width: int, window_size: int):
    """
    Does all the preprocessing neccessary to convert the given audio file to a spectogram
    the model can work with
    :param audio_file:
    :return:
    """
    audio_windows = split_into_windows(audio_file=audio_file, sr=sr, window_size=window_size)
    spectogram_list = [get_spectogram(audio, sr=sr, n_mels=n_mels, spec_height=height, spec_width=width) for audio in audio_windows]

    return torch.stack(spectogram_list, dim=0)
def preprocess_raw_audio(audio_file):
    pass

def preprocess_mfcc(audio_file):
    pass


def get_spectogram(audio_file: np.ndarray, sr: int, n_mels: int, spec_height: int, spec_width: int) -> torch.Tensor:
    """
    Constructs a spectogram tensor using the given audio signal and, optionally, augments it.

    :param audio_file: audio signal used to create the spectogram
    :param sr: sampling rate of the audio signal
    :param n_mels: number of Mel bands to generate
    :param spec_height: height to which to resize the spectogram
    :param spec_width: width to which to resize the spectogram
    :param augmentation: indicates whether to use frequency and time masking augmentation on the spectogram
    :return: spectogram tensor
    """
    spectogram = lr.feature.melspectrogram(y=audio_file, sr=sr, n_mels=n_mels)

    spectogram = spectogram_to_db(spectogram)
    resized_spectogram = resize(spectogram, (spec_height, spec_width))

    spectogram_tensor = torch.from_numpy(resized_spectogram).float()
    spectogram_tensor = torch.unsqueeze(spectogram_tensor, 0)

    # ovaj repeat je samo da dobijemo rgb sliku iz 1-kanalnog spektograma
    # pa ponavljamo samo 1 kanal 3x
    return spectogram_tensor.repeat(3, 1, 1)

def spectogram_to_db(spectogram: np.ndarray) -> np.ndarray:
    """
    Wrapper around the lr.power_to_db method.
    """
    return lr.power_to_db(spectogram, ref=np.max)


