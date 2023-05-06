import ast
import logging
import os
import re
from collections import defaultdict

import ffmpeg
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.classification import MultilabelAccuracy
from tqdm import tqdm
from transformers import AutoFeatureExtractor

# TODO: ovo sve dodati u hydra config!
genres = ["[cou_fol]", "[cla]", "[pop_roc]", "[lat_sou]", "[jaz_blu]"]

data_root_dir = "../../../../Dataset"
datalists_dir = "../../../../Dataset/datalists"

train_prefix = "IRMAS_Training_Data"
test_prefix = "IRMAS_Validation_Data"

class_mappings = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}
class_mappings_inv = {v: k for k, v in class_mappings.items()}

VAL_PERCENTAGE = 0.3

NO_CLASSES = 11

def walk_directory_train_data(root_dir):
    file_info = []
    for class_folder in os.listdir(root_dir):
        class_folder_path = os.path.join(root_dir, class_folder)
        class_id = class_mappings[class_folder]

        for file_name in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, file_name)

            #abs_file_path = os.path.abspath(file_path)

            audio_length = lr.get_duration(filename=file_path)
            genre = None
            drums = None
            for g in genres:
                if (g in file_name):
                    genre = g[1:-1]
                    break
            if ("[dru]" in file_name):
                drums = True
            elif ("[nod]" in file_name):
                drums = False

            file_info.append({
                "classes": [class_folder],
                "classes_id": [class_id],
                "file_name": file_name,
                "file_path": train_prefix + "/" + class_folder + "/" + file_name,
                "audio_length": audio_length,
                "genre": genre,
                "drums": drums
            })
    # create a pandas data frame from the file information list
    return pd.DataFrame(file_info)


def walk_directory_test_data(root_dir):
    file_info = []
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)

        root_ext = os.path.splitext(file_path)
        root_file_name = root_ext[0]
        ext = root_ext[1]

        if (ext == ".txt"):
            continue

        wav_file_path = file_path
        txt_file_path = root_file_name + ".txt"

        classes = []
        with open(txt_file_path, "r") as f:
            classes = f.read().replace("\t", "").split("\n")
            classes = [c for c in classes if c != ""]

        classes_id = [class_mappings[c] for c in classes]

        #abs_file_path = os.path.abspath(wav_file_path)

        audio_length = lr.get_duration(filename=wav_file_path)

        file_info.append({
            "classes": classes,
            "classes_id": classes_id,
            "file_name": file,
            "file_path": test_prefix + "/" + file,
            "audio_length": audio_length
        })

    return pd.DataFrame(file_info)

def test_val_split():
    # Load the data from the .csv file
    data = pd.read_csv(os.path.join(datalists_dir, 'test_original.csv'))
    test_data, val_data = train_test_split(data, test_size=VAL_PERCENTAGE)
    #test_data = test_data.drop('Unnamed: 0', axis=1)
    test_data.to_csv(os.path.join(datalists_dir, 'test.csv'), index=False)
    val_data.to_csv(os.path.join(datalists_dir, 'val.csv'), index=False)


#Number of records: 6705
#Split: train; Mean: -0.000404580, std. deviation: 0.108187131

#Number of records: 2874
#Split: test; Mean: -0.000190258, std. deviation: 0.131417455
def calculate_mean_and_std_deviation(csv_path, target_sr):
    df = pd.read_csv(csv_path)
    df_dict = df.to_dict('records')
    mean_sum = 0
    std_dev_sum = 0
    count = 0
    for row in tqdm(df_dict):
        file_path = row['file_path']
        try:
            # if file path is absolute (audioset), data_root_dir is discarded
            # if file path is relative (IRMAS), data_root_dir is used
            y, _ = lr.load(os.path.join(data_root_dir, file_path), sr=target_sr)
        except Exception as e:
            print(f"Error: {e}")
            print(f"File path: {file_path}")
            continue
        mean_sum += np.mean(y)
        std_dev_sum += np.std(y)
        count += 1

    mean_sum /= count
    std_dev_sum /= count

    print('Number of records:', count)

    print(f"CSV file: {csv_path}; Mean: {mean_sum:.9f}, std. deviation: {std_dev_sum:.9f}")


def calculate_mean_and_std_mel_spectrogram(csv_path, target_sr, audio_mean, audio_std_dev, max_length):
    featurizer = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=max_length)
    df = pd.read_csv(csv_path)
    df_dict = df.to_dict('records')
    means = []
    std_devs = []
    for row in tqdm(df_dict):
        file_path = row['file_path']
        try:
            # if file path is absolute (audioset), data_root_dir is discarded
            # if file path is relative (IRMAS), data_root_dir is used
            y, _ = lr.load(os.path.join(data_root_dir, file_path), sr=target_sr)
            y = (y - audio_mean) / audio_std_dev
            mel_spectrogram = featurizer(y, return_tensors="pt", sampling_rate=featurizer.sampling_rate).input_values
        except Exception as e:
            print(f"Error: {e}")
            print(f"File path: {file_path}")
            continue
        means.append(torch.mean(mel_spectrogram))
        std_devs.append(torch.std(mel_spectrogram))

    mean = np.mean(means)
    std_dev = np.mean(std_devs)

    print("Number of records:", len(means))

    print(f"CSV file: {csv_path}; Mean: {mean:.9f}, std. deviation: {std_dev:.9f}")


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
        'hamming_score': 1-hamming_loss(target, pred)
    }

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i=0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i=i+1
    print('-----------------------------------------------')

# To save the checkpoint
def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def train_val_test_split(data, train_percentage=0.7, val_percentage=0.1, test_percentage=0.2, seed=42):
    assert train_percentage + val_percentage + test_percentage == 1, "Train, val and test percentages must sum to 1"
    train_data, test_data = train_test_split(data, test_size=test_percentage, random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=val_percentage / (train_percentage + val_percentage), random_state=seed)
    return train_data, val_data, test_data

def split_dataset(csv_path, dst_path, train_percentage=0.7, val_percentage=0.1, test_percentage=0.2, seed=42, columns_to_keep=["file_path", "classes_id"]):
    data = pd.read_csv(csv_path, usecols=columns_to_keep)
    train_data, val_data, test_data = train_val_test_split(data, train_percentage, val_percentage, test_percentage, seed)
    train_data.to_csv(os.path.join(dst_path, "train.csv"), index=False)
    val_data.to_csv(os.path.join(dst_path, "val.csv"), index=False)
    test_data.to_csv(os.path.join(dst_path, "test.csv"), index=False)

def change_mp3_to_wav(csv_file, dst_path):
    new_df_dict = {"file_path": [], "classes_id": [], "classes": [], "YTID": [], "start_second": [], "end_second": [], "positive_labels": []}
    df = pd.read_csv(csv_file)
    df_dict = df.to_dict('records')
    exception_count = 0
    for i, row in tqdm(enumerate(df_dict)):
        file_path = row['file_path']
        file_name = file_path.split('/')[-1].split('.mp3')[0]
        # some problems if \" was also removed
        file_name = re.sub(r'([.!?<>:|*\/\\]|\s+)', '_', file_name)
        new_path = os.path.join(dst_path, file_name + '.wav')
        if os.path.exists(new_path):
            new_path = os.path.join(dst_path, file_name + '_' + str(i) + '.wav')
        try:
            ffmpeg.input(file_path).output(new_path, acodec='pcm_s16le', ac=1, ar=44100, f='wav').run(quiet=True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"File path: {file_path}")
            exception_count += 1
            continue
        new_df_dict['file_path'].append(new_path)
        new_df_dict['classes_id'].append(row['classes_id'])
        new_df_dict['classes'].append(row["classes"])
        new_df_dict['YTID'].append(row["YTID"])
        new_df_dict['start_second'].append(row["start_second"])
        new_df_dict['end_second'].append(row["end_second"])
        new_df_dict['positive_labels'].append(row["positive_labels"])
    print(f"Number of exceptions: {exception_count}")
    new_df = pd.DataFrame.from_dict(new_df_dict)
    new_df.to_csv(os.path.join(dst_path, "audioset_wav.csv"), index=False)


def align_audio_lengths(csv_file, sr=44100, audio_length=10, threshold_in_seconds=0.5):
    df = pd.read_csv(csv_file)
    dictionary = df.to_dict(orient="records")
    # new dictionary to store the new data
    new_dict = {"file_path": [], "classes_id": [], "classes": [], "YTID": [], "start_second": [], "end_second": [], "positive_labels": []}
    counter = 0
    threshold = int(sr * threshold_in_seconds)
    expected_length = int(sr * audio_length)
    for row in tqdm(dictionary):
        audio_file_path = row["file_path"]
        audio_file, sr = lr.load(audio_file_path, sr=44100)
        if abs(len(audio_file) - expected_length) > threshold:
            counter += 1
            # delete the file
            os.remove(audio_file_path)
        else:
            # use librosa.util.fix_length to make the audio file have the expected length
            audio_file = lr.util.fix_length(audio_file, size=expected_length)
            try:
                sf.write(file=audio_file_path, data=audio_file, samplerate=sr)
            except Exception as e:
                print(f"Error: {e}")
                print(f"File path: {audio_file_path}")
                continue
            new_dict["file_path"].append(audio_file_path)
            new_dict["classes_id"].append(row["classes_id"])
            new_dict["classes"].append(row["classes"])
            new_dict["YTID"].append(row["YTID"])
            new_dict["start_second"].append(row["start_second"])
            new_dict["end_second"].append(row["end_second"])
            new_dict["positive_labels"].append(row["positive_labels"])

    print(f"From the total of {len(dictionary)} files, {counter} files ({counter / len(dictionary)} %) have a corrupted length when using threshold {threshold} ({threshold / 44100} seconds).")

    new_df = pd.DataFrame.from_dict(new_dict)
    new_df.to_csv("../../../../Dataset/audioset/audioset_wav_fixed_lengths.csv", index=False)


def analyze_dataset(csv_path: str, sr: int, dataset_name: str) -> None:
    assert dataset_name in ["IRMAS", "audioset"], "Dataset name must be either IRMAS or audioset."
    split_name = csv_path.split("/")[-1].split(".")[0]
    durations = []
    labels_list = []
    instances_per_class = defaultdict(int)
    durations_by_class = defaultdict(list)
    number_of_labels_per_instance = []
    df = pd.read_csv(csv_path)
    dictionary = df.to_dict(orient="records")
    for row in tqdm(dictionary):
        audio_file_path = os.path.join(data_root_dir, row["file_path"])
        audio_file, sr = lr.load(audio_file_path, sr=sr)
        labels = ast.literal_eval(row["classes_id"])
        duration = len(audio_file) / sr
        durations.append(duration)
        labels_list.append(labels)
        number_of_labels_per_instance.append(len(labels))
        for label in labels:
            instances_per_class[label] += 1
            durations_by_class[label].append(duration)

    os.makedirs(f"./plots/{dataset_name}/{split_name}", exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set(font_scale=2)

    # plot the durations using seaborn
    sns.displot(durations, kde=True, height=10, aspect=2)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of instances")
    plt.savefig(f"./plots/{dataset_name}/{split_name}/duration_distribution.png")
    plt.show()

    # co-occurrence matrix from labels_list
    mlb = MultiLabelBinarizer(classes=range(NO_CLASSES))
    one_hot_labels = mlb.fit_transform(labels_list)
    co_occurrence_matrix = np.dot(one_hot_labels.T, one_hot_labels)
    plt.figure(figsize=(21, 20))
    sns.heatmap(co_occurrence_matrix, xticklabels=class_mappings.keys(), yticklabels=class_mappings.keys(), cmap="Blues")
    plt.title("Label co-occurrence matrix")
    plt.savefig(f"./plots/{dataset_name}/{split_name}/co_occurrence_matrix.png")
    plt.show()

    # plot the number of labels per instance as a bar plot
    plt.figure(figsize=(20, 10))
    sns.countplot(y=sorted(number_of_labels_per_instance), color="b")
    plt.xlabel("Number of instances")
    plt.ylabel("Number of labels")
    plt.savefig(f"./plots/{dataset_name}/{split_name}/number_of_labels_per_instance.png")
    plt.show()

    # plot the number of instances per class sorted by number of instances in a descent order; plot classes on y axis and number of instances on x axis
    instances_per_class = {k: v for k, v in sorted(instances_per_class.items(), key=lambda item: item[0])}
    instances_per_class = {class_mappings_inv[k]: v for k, v in instances_per_class.items()}
    plt.figure(figsize=(16, 20))
    sns.barplot(x=[instances_per_class[i] for i in instances_per_class.keys()], y=[i for i in instances_per_class.keys()], order=[class_mappings_inv[elem] for elem in np.argsort([-instances_per_class[i] for i in instances_per_class.keys()])], color="b", orient='h')
    plt.xlabel("Number of instances")
    plt.ylabel("Class")
    plt.savefig(f"./plots/{dataset_name}/{split_name}/instances_per_class.png")
    plt.show()

    # plot the durations per class as violin plots
    durations_by_class = {class_mappings_inv[k]: durations_by_class[k] for k in sorted(durations_by_class)}
    plt.figure(figsize=(16, 20))
    sns.violinplot(data=list(durations_by_class.values()), inner="stick", orient="h", cut=0)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Class")
    plt.yticks([i for i in range(NO_CLASSES)], [class_mappings_inv[i] for i in range(NO_CLASSES)])
    plt.savefig(f"./plots/{dataset_name}/{split_name}/durations_per_class.png")
    plt.show()

    # plot duration (float) and number of labels per instance (int) as a violin plot
    plt.figure(figsize=(20, 15))
    sns.violinplot(x=durations, y=number_of_labels_per_instance, cut=0, inner="stick", orient="h")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of labels")
    plt.savefig(f"./plots/{dataset_name}/{split_name}/durations_per_num_labels.png")
    plt.show()
