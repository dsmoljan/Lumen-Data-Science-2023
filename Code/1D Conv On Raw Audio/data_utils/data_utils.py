from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
import os
import pandas as pd
import librosa
import torch

# izracunaj mean i std. deviaciju svog dataseta i to koristi https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html#:~:text=Normalizing%20the%20image%20dataset%20means,by%20the%20channel%20standard%20deviation.
# ne koristiti ovo 0.5, 0.5, 0.5, to je samo neka dummy vrijednost
def get_transformation(size, resize=False):
    transform_list = [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize([.5, .5, .5], [.5, .5, .5])
    ]
    return Compose(transform_list)

def calculate_mean_deviation(dataloader):
    mean = 0.0
    std = 0.0

    no_samples = 0

    for images in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        no_samples += batch_samples

    mean /= no_samples
    std /= no_samples

    print(f"Mean: {mean}, std. deviation: {std}")

def walk_directory_train_data(root_dir):
    file_info = []
    for class_folder in os.listdir(root_dir):
        class_folder_path = os.path.join(root_dir, class_folder)

        for file_name in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, file_name)

            audio_length = librosa.get_duration(filename=file_path)

            file_info.append({
                "class": class_folder,
                "file_name": file_name,
                "file_path": file_path,
                "audio_length": audio_length
            })
    # create a pandas data frame from the file information list
    df = pd.DataFrame(file_info)

def walk_directory_test_data(root_dir):
    pass

def collate_fn_windows(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    examples, labels, lengths = zip(*data)
    max_len = max(lengths)
    example_shape = examples[0][0].size()
    # features shape should be [batch_size, max_window_num, 3, 128, 128] for spectograms
    # and for audio, their shape should be [batch_size, max_window_num, SR*window_size]
    # ovo je sad ok za spektograme, ali trebaš prilagoditi kod t.d radi i za audio
    # * operator unpacks tuple to an array of ints
    features = torch.zeros((len(data), max_len, *example_shape))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    # TODO: tu si stao, trebas dovrsiti
    # mislim da je ovo napisano za 1D tenzore, trebati će ih prilagoditi za trokanalne tenzore
    for i in range(len(data)):
        j = lengths[i]
        emtpy_tensor = torch.zeros((max_len - j, *example_shape))
        features[i] = torch.cat((torch.stack(examples[i]), emtpy_tensor))

    return features.float(), labels.long(), lengths.long()