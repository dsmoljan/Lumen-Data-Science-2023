import os
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa as lr
import numpy as np
from torchmetrics.classification import MultilabelAccuracy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import logging

from pytorch_lightning.utilities import rank_zero_only

# TODO: ovo sve dodati u hydra config!
genres = ["[cou_fol]", "[cla]", "[pop_roc]", "[lat_sou]", "[jaz_blu]"]

data_root_dir = "../../../../Dataset"
datalists_dir = "../../../../Dataset/datalists"

train_prefix = "IRMAS_Training_Data"
test_prefix = "IRMAS_Validation_Data"

class_mappings = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}

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
def calculate_mean_and_std_deviation(target_sr, split):
    assert split in ["train", "test"], "Split must be either 'train' or 'test'"
    df = pd.read_csv(os.path.join(datalists_dir, "test_original.csv")) if split == "test" else pd.read_csv(os.path.join(datalists_dir, "train.csv"))
    df_dict = df.to_dict('records')
    mean_sum = 0
    std_dev_sum = 0
    count = 0
    for row in df_dict:
        path = row["file_path"]
        file_path = os.path.join(data_root_dir, path)
        y,_ = lr.load(file_path, sr=target_sr)
        mean_sum += np.mean(y)
        std_dev_sum += np.std(y)
        count += 1

    mean_sum /= count
    std_dev_sum /= count

    print('Number of records:', count)

    print(f"Split: {split}; Mean: {mean_sum:.9f}, std. deviation: {std_dev_sum:.9f}")

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
    }

import torch

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