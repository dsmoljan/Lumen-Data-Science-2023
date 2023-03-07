import os
import pandas as pd
import librosa

df = None

genres = ["[cou_fol]", "[cla]", "[pop_roc]", "[lat_sou]", "[jaz_blu]"]

train_root_dir = "../../../Dataset/IRMAS_Training_Data"
test_root_dir = "../../../Dataset/IRMAS_Validation_Data"

def walk_directory_train_data(root_dir):
    file_info = []
    for class_folder in os.listdir(root_dir):
        class_folder_path = os.path.join(root_dir, class_folder)

        for file_name in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, file_name)

            abs_file_path = os.path.abspath(file_path)

            audio_length = librosa.get_duration(filename=file_path)
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
                "class": class_folder,
                "file_name": file_name,
                "file_path": abs_file_path,
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

        abs_file_path = os.path.abspath(wav_file_path)

        audio_length = librosa.get_duration(filename=wav_file_path)

        file_info.append({
            "classes": classes,
            "file_name": file,
            "file_path": abs_file_path,
            "audio_length": audio_length
        })

    return pd.DataFrame(file_info)


def main():
    print("Starting directory walking util.")

    df_train = walk_directory_train_data(train_root_dir)
    df_test = walk_directory_test_data(test_root_dir)
    df_train.to_csv("train.csv")
    df_test.to_csv("test.csv")

    print("Train and test .csv files generated")


main()