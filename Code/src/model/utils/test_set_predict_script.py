# a small script which iterates over all examples in the test set
# sends them to the prediction endpoint, gets a prediction
# and creates a .json file with predictions for every example
import json
import os

import requests
from tqdm import tqdm

ENDPOINT_URL = "http://localhost:8080/api/predict"
FILE = "(02) dont kill the whale-1.wav"
DIRECTORY = "C:/Users/dsmoljan/Desktop/Lumen natjecanje/test_dataset-20230428T085441Z-001/test_dataset"
def main():
    result_dict = {}
    #ignal, sampling_rate = audiofile.read(FILE)
    for filename in tqdm(os.listdir(DIRECTORY)):
        if filename.endswith('.json'):
            continue
        f = os.path.join(DIRECTORY, filename)
        # checking if it is a file
        filename = filename.replace(".wav", "")
        if os.path.isfile(f):
            with open(f, mode='rb') as file:  # b is important -> binary
                fileContent = file.read()
                response = requests.post(ENDPOINT_URL, files={"audio_file": fileContent})
                result_dict.update({filename: response.json()})

    with open("test_results.json", "w") as outfile:
        json.dump(result_dict, outfile)
    json_object = json.dumps(result_dict)
    #song = AudioSegment.from_wav(FILE)
    #return response.json()

if __name__ == "__main__":
    main()