import torch
import numpy as np

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
    features = torch.zeros((len(data), max_len, *example_shape))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j = lengths[i]
        emtpy_tensor = torch.zeros((max_len - j, *example_shape))
        features[i] = torch.cat((torch.stack(examples[i]), emtpy_tensor))

    return features.float(), labels.long(), lengths.long()
