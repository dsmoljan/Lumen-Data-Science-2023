from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
import os
import pandas as pd
import librosa

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