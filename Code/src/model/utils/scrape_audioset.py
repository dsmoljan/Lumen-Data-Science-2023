import os
import time
from collections import Counter

import ffmpeg
import pandas as pd
from pytube import YouTube
from tqdm import tqdm

class_mappings = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}

audioset_id_to_class = {
    "/m/01xqw": "cel",
    "/m/01wy6": "cla",
    "/m/0l14j_": "flu",
    "/m/042v_gx": "gac",
    "/m/02sgy": "gel",
    "/m/013y1f": "org", # maybe remove hammond/electric organ
    "/m/05r5c": "pia", # includes electric piano
    "/m/06ncr": "sax", # include both alto and soprano saxophone, good
    "/m/07gql": "tru",
    "/m/07y_7": "vio", # REMOVE PIZZICATO ("/m/0d8_n")
    "/m/02qldy": "voi",  # narration
    "/m/01h8n0": "voi",  # conversation
    "/m/02zsn": "voi",  # female speech
    "/m/05zppz": "voi",  # male speech
    "/t/dd00005": "voi",  # child singing
    "/t/dd00004": "voi",  # female singing
    "/t/dd00003": "voi",  # male singing
    "/m/0l14jd": "voi",  # choir
}

MAX_AUDIOSET_NUM_FILES_PER_CLASS = 2000

def download_mp3_from_ytid(ytid, start_second, end_second, output_dir='./'):
    link = "https://www.youtube.com/watch?v=" + ytid
    yt = YouTube(link)

    try:
        # if file already exists, skip download
        if os.path.isfile(os.path.join(output_dir, yt.title.replace("/", "-") + ".mp3")):
            return os.path.join(output_dir, yt.title.replace("/", "-") + ".mp3")

        print(f"[*] Downloading video with ID {ytid}")
        name = yt.title.replace("/", "-") + "(uncut)" + ".mp3"
        yt.streams.filter(only_audio=True).first().download(output_path=output_dir, filename=name, skip_existing=False)

        audio_input = ffmpeg.input(os.path.join(output_dir, name))
        audio_output = ffmpeg.output(audio_input, os.path.join(output_dir, yt.title.replace("/", "-") + ".mp3"), ss=start_second, to=end_second)

        ffmpeg.run(audio_output, quiet=True, overwrite_output=True)
        os.remove(os.path.join(output_dir, name))
        return os.path.join(output_dir, yt.title.replace("/", "-") + ".mp3")

    except Exception as e:
        class_name = e.__class__.__name__
        if class_name == "URLError":
            print("[**********] URLError caught: " + str(e.reason))
            time.sleep(60)
        else:
            print(class_name + ": " + str(e))
        return None

def download_audioset_yt_data(csv_file_data, scraped_file_path, output_dir='./', continue_from=None):
    #df_class_label_indices = pd.read_csv(csv_file_class_label_indices)
    #class_label_indices = df_class_label_indices.set_index('mid').T.to_dict('records')[0]  # format: {'/m/09x0r': 'Speech'}
    ytids = set()
    if continue_from is None:
        data = pd.read_csv(csv_file_data, quotechar='"', header=0, sep=", ", engine='python')
        class_counters = Counter()
    else:
        print(f"Continuing from row: {continue_from}...")
        data = pd.read_csv(csv_file_data, quotechar='"', header=None, names=["YTID", "start_seconds", "end_seconds", "positive_labels"], sep=", ", engine='python', skiprows=continue_from) # skip the header row (at least)
        class_counters = Counter()
        scraped_file = pd.read_csv(scraped_file_path)
        print("Loading previous class counters...")
        for _, row in scraped_file.iterrows():
            for c in eval(row["classes"]):
                class_counters[c] += 1
            ytids.add(row["YTID"])
        print(class_counters)
    file_info = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        i += continue_from if continue_from is not None else 0

        # checkpointing
        if (i % 1000 == 0):
            print("i: ", i)
            print(class_counters)
            print("\n")
            # if continue_from is None, save to a csv file, otherwise append to the existing csv file
            if (continue_from is None):
                pd.DataFrame(file_info).to_csv(scraped_file_path, index=False)
            else:
                pd.DataFrame(file_info).to_csv(scraped_file_path, mode='a', header=False, index=False)
                file_info = []

        if (i % 2000 == 0 and i != 0):
            # prevent from being blocked by youtube
            print("\nSleeping for 30 seconds...\n")
            time.sleep(30)

        # if the YTID is already in the scraped file, skip it
        if row["YTID"] in ytids:
            continue

        # eval is very important here, otherwise the string will be read incorrectly
        positive_labels = eval(row["positive_labels"]).split(",")

        # only keep keep those we are interested in
        positive_labels = [label.strip() for label in positive_labels if label in audioset_id_to_class]

        # if both violin and pizzicato are present, remove violin
        if ("/m/07y_7" in positive_labels and "/m/0d8_n" in positive_labels):
            positive_labels.remove("/m/07y_7")

        # if both piano and electric piano are present, remove piano
        if ("/m/05r5c" in positive_labels and "/m/01s0ps" in positive_labels):
            positive_labels.remove("/m/05r5c")

        # if there are no labels we are interested in, continue
        if (len(positive_labels) == 0):
            continue

        classes = [audioset_id_to_class[label] for label in positive_labels]

        # check if "voi" appears in classes multiple times, keep only one
        while (classes.count("voi") > 1):
            classes.remove("voi")

        classes_id = [class_mappings[c] for c in classes]

        # continue only if there is a class in classes which is present in the counter less than MAX_NUM_FILES_PER_CLASS times
        if (not any([class_counters[c] < MAX_AUDIOSET_NUM_FILES_PER_CLASS for c in classes])):
            continue

        ytid = row["YTID"]
        start_second = row["start_seconds"]
        end_second = row["end_seconds"]
        path = download_mp3_from_ytid(ytid, start_second, end_second, output_dir)
        if (path == None):
            continue

        file_info.append({
            "YTID": ytid,
            "path": path,
            "classes": classes,
            "classes_id": classes_id,
            "start_second": start_second,
            "end_second": end_second,
            "positive_labels": ",".join(positive_labels)
        })
        # update the class counters
        for c in classes:
            class_counters[c] += 1

    # save the file info
    print(class_counters)
    # if continue_from is None, save to a csv file, other wise append to the existing csv file
    if (continue_from is None):
        pd.DataFrame(file_info).to_csv(scraped_file_path, index=False)
    else:
        pd.DataFrame(file_info).to_csv(scraped_file_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    download_audioset_yt_data(
        csv_file_data="../../../../Dataset/audioset/csv_files/unbalanced_train_segments.csv",
        scraped_file_path="../../../../Dataset/audioset/csv_files/audioset_scraped.csv",
        output_dir="../../../../Dataset/audioset/audio_files_final/",
        continue_from=1
    )