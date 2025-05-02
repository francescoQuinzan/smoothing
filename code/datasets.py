from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

import device
from device import DEVICE

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

def denormalize_images(dataset: str, images) :
    """Return the dataset as a PyTorch Dataset object"""

    if dataset == "imagenet":
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor(_IMAGENET_STDDEV).view(1, 3, 1, 1).to(DEVICE)

    elif dataset == "cifar10":
        mean = torch.tensor(_CIFAR10_MEAN).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor(_CIFAR10_STDDEV).view(1, 3, 1, 1).to(DEVICE)

    images = images.to(DEVICE)
    denorm_images = images * std + mean
    denorm_images = denorm_images.clone().detach().requires_grad_(True)
    return denorm_images
    
def normalize_images(dataset: str,images) :
    """Return the dataset as a PyTorch Dataset object"""

    if dataset == "imagenet":
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor(_IMAGENET_STDDEV).view(1, 3, 1, 1).to(DEVICE)

    elif dataset == "cifar10":
        mean = torch.tensor(_CIFAR10_MEAN).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor(_CIFAR10_STDDEV).view(1, 3, 1, 1).to(DEVICE)

    images = images.to(DEVICE)
    norm_images = (images - mean) / std
    norm_images = norm_images.clone().detach().requires_grad_(True)
    return norm_images

def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    
def get_dimensions(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return True # I don't know what is the dimension of imagenet in this case
    elif dataset == "cifar10":
        return 32 * 32


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./datasets/cifar10", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./datasets/cifar10", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """

        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(DEVICE)
        self.sds = torch.tensor(sds).to(DEVICE)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
