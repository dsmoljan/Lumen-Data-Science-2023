import os

from utils import (
    align_audio_lengths,
    calculate_mean_and_std_deviation,
    change_mp3_to_wav,
    split_dataset,
    test_val_split,
    walk_directory_test_data,
    walk_directory_train_data,
)

train_root_dir = "../../../../Dataset/IRMAS_Training_Data"
test_root_dir = "../../../../Dataset/IRMAS_Validation_Data"
datalists_dir = "../../../../Dataset/datalists"

def main(dataset="IRMAS"):
    assert dataset in ["IRMAS", "audioset"], "Dataset must be either IRMAS or audioset."

    if dataset == "IRMAS":
        print("Starting directory walking util.")

        df_train = walk_directory_train_data(train_root_dir)
        df_test = walk_directory_test_data(test_root_dir)
        df_train.to_csv(os.path.join(datalists_dir, "../../Util/train.csv"))
        df_test.to_csv(os.path.join(datalists_dir, "../../Util/test_original.csv"))

        print("Train and test .csv files generated")

        calculate_mean_and_std_deviation(os.path.join(datalists_dir, "../../Util/train.csv"), 44100)
        calculate_mean_and_std_deviation(os.path.join(datalists_dir, "../../Util/test_original.csv"), 44100)

        test_val_split()
    else:
        csv_file = "../../../../Dataset/audioset/csv_files/audioset_scraped.csv"
        dst_path = "../../../../Dataset/audioset/wav/"

        change_mp3_to_wav(csv_file, dst_path)
        new_csv = os.path.join(dst_path, "..", "audioset_wav.csv")
        align_audio_lengths(new_csv, sr=44100, audio_length=10, threshold_in_seconds=0.5)
        new_csv = os.path.join(dst_path, "..", "audioset_wav_fixed_lengths.csv")
        split_dataset(new_csv, os.path.join(dst_path, "..", "datalists/"))
        calculate_mean_and_std_deviation(os.path.join(dst_path, "..", "datalists/", "train.csv"), 44100)

main(dataset="IRMAS")
