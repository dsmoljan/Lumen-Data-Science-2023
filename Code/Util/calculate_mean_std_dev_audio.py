import os

import numpy as np
import pandas as pd
import librosa as lr

train_root_dir = "../../../Dataset/"

def calculate_mean_and_std_deviation(target_sr):
    df = pd.read_csv("train.csv")
    df_dict = df.to_dict('records')
    mean_sum = 0
    std_dev_sum = 0
    count = 0
    for row in df_dict:
        path = row["file_path"]
        file_path = os.path.join(train_root_dir, path)
        y,sr = lr.load(file_path, target_sr)
        mean_sum += np.mean(y)
        std_dev_sum += np.std(y)
        count += 1

    mean_sum /= count
    std_dev_sum /= count

    print(f"Mean: {mean_sum:.6f}, std. deviation: {std_dev_sum:.6f}")


def main():
    calculate_mean_and_std_deviation(44100)

main()