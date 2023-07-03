# Lumen-Data-Science-2023  
Team: Harmony Is All You Need  
Members: Dorian Smoljan, Tin Ferković  
3rd place winner

Repository for the Lumen Data Science 2023 competition.

Here you can find our solution to the problem of multi-label audio classification. This [task](https://drive.google.com/file/d/16SPgrFzO6uFc0Za-gjOOY9zCUF7shMXX/view) was the topic of the 2023 competition organized by e-student.
Project documentation can be found [here](https://github.com/dsmoljan/Lumen-Data-Science-2023/blob/main/Documentation/Project%20documentation/Paper/Lumen%20Data%20Science%20project%20documentation%2C%20Harmony%20Is%20All%20You%20Need.pdf) and technical documentation [here](https://github.com/dsmoljan/Lumen-Data-Science-2023/blob/main/Documentation/Technical%20documentation/Paper/Lumen%20Data%20Science%20technical%20documentation%2C%20Harmony%20Is%20All%20You%20Need.pdf).

If you want to build onto this project, follow these instructions:
1. After cloning the project, position yourself into `Code/src/model/` directory.
2. Run `conda env create -f environment.yaml` to create a conda environment.
3. Activate the environment using `conda activate lumen`
4. Build on top of it.

If you want to run the application, follow the instructions provided in the [technical documentation](https://github.com/dsmoljan/Lumen-Data-Science-2023/blob/main/Documentation/Technical%20documentation/Paper/Lumen%20Data%20Science%20technical%20documentation%2C%20Harmony%20Is%20All%20You%20Need.pdf). You will need a valid checkpoint to be able to make any audio predictions. In that case, feel free to contact the contributors of this project.

All the hydra configs are provided in the [configs](https://github.com/dsmoljan/Lumen-Data-Science-2023/tree/main/Code/configs) folder in order to make (hyper)parameter changes easier.
The base class for the models is [audio_model_lightning.py](https://github.com/dsmoljan/Lumen-Data-Science-2023/blob/main/Code/src/model/models/audio_model_lightning.py) in case of a single optimizer for both base model's and classification head's parameters or [audio_model_separate_optimizers.py](https://github.com/dsmoljan/Lumen-Data-Science-2023/blob/main/Code/src/model/models/audio_model_lightning_seperate_optimizers.py) in case of seperate learning rates (and optimizers) for these two groups of parameters.
All the components we've built (ResNet, Audio Spectrogram Transformer, and various simpler CNN architectures) are placed in the [components](https://github.com/dsmoljan/Lumen-Data-Science-2023/tree/main/Code/src/model/models/components) folder and you can add your own components accordingly as well.

For any further questions, we refer the reader to the technical or project documentation. If these do not answer them, feel free to contact the contributors.
