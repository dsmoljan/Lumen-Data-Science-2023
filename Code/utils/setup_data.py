import os
from utils import walk_directory_test_data, walk_directory_train_data, calculate_mean_and_std_deviation, test_val_split
train_root_dir = "../../../Dataset/IRMAS_Training_Data"
test_root_dir = "../../../Dataset/IRMAS_Validation_Data"
datalists_dir = "../../../Dataset/datalists"

def main():
    print("Starting directory walking util.")

    df_train = walk_directory_train_data(train_root_dir)
    df_test = walk_directory_test_data(test_root_dir)
    df_train.to_csv(os.path.join(datalists_dir, "train.csv"))
    df_test.to_csv(os.path.join(datalists_dir, "test_original.csv"))

    print("Train and test .csv files generated")

    calculate_mean_and_std_deviation(44100, "train")
    calculate_mean_and_std_deviation(44100, "test")

    test_val_split()

main()