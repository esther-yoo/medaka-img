from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset
import pandas as pd
import skimage as ski
import os


class MinMaxScaling:
    """
    Custom min-max scaling function to pass into Compose.
    Transforms pixel values to range [0,1] for each image individually.

    Methods:
        __call__(tensor): Scales the input tensor to the range [0, 1].
    """

    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class MedakaDataset(Dataset):
    """
    Custom dataset class for the medaka fish images.

    Attributes:
        image_paths (pd.Series):
            Pandas Series containing the paths to the images.
        src_dir (str):
            Directory containing the images.
        transform (Compose):
            Compose object containing the transformations to apply.
        direction_csv (str):
            Path to the CSV file containing whether the fish
            are facing left or right.
    """

    def __init__(self, data_csv, direction_csv, src_dir, transform, config):
        self.image_paths = pd.read_csv(data_csv)['img_name']
        self.src_dir = src_dir
        self.transform = transform
        self.config = config

        # Load and process the CSV file containing the direction of the fish
        self.direction_csv = pd.read_csv(direction_csv)
        self.direction_csv = self.direction_csv[
            self.direction_csv['left_facing'] == 'left']
        self.direction_csv = self.direction_csv.rename(
            columns={'Unnamed: 0': 'img_name'})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = f'{self.src_dir}/{img_name}'
        
        if self.config['model'] in ["vanilla-ae", "vanilla-ae-relu"]:
            image = ski.io.imread(img_path)[:, :, 0]
        else:
            print("Loading image")
            image = ski.io.imread(img_path)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # If facing left, flip the image horizontally
        img_base_name = os.path.splitext(img_name)[0]
        if img_base_name in self.direction_csv['img_name'].values:
            image = transforms.functional.hflip(image)

        return image, img_name


def transform(resize_shape=(224, 224)):
    """
    Returns a Compose object that applies the following transformations:
        1. Convert the image to a tensor.
        2. Crop out the bottom text.
        3. Resize the image to desired dimensions (default: 244x244).
        4. Normalize the pixel values to the range [0, 1].

    Returns:
        Compose: A Compose object that applies the specified transformations.
    """

    transformations = [
        transforms.ToTensor(),
        transforms.Lambda(
            lambda img: transforms.functional.crop(img, 0, 0, 980, 1392)),
        transforms.Resize(resize_shape),
        MinMaxScaling()
    ]

    return transforms.Compose(transformations)
