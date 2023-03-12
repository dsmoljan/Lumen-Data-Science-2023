import torch
import librosa
import numpy
import matplotlib as plt
audio_file = "../../../Dataset/IRMAS_Training_Data/cel/[cel][cla]0001__1.wav"

y, sr = librosa.load(audio_file)
D = librosa.stft(y)
H, P = librosa.decompose.hpss(D)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

plt.show(S)