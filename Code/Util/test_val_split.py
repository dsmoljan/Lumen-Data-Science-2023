import pandas as pd
from sklearn.model_selection import train_test_split

VAL_PERCENTAGE = 0.3


def test_val_split():
    # Load the data from the .csv file
    data = pd.read_csv('test_original.csv')
    test_data, val_data = train_test_split(data, test_size=VAL_PERCENTAGE)
    #test_data = test_data.drop('Unnamed: 0', axis=1)
    test_data.to_csv('test_new.csv', index=False)
    val_data.to_csv('val_new.csv', index=False)


def main():
    test_val_split()

main()
