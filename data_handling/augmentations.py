import torch
import torchvision.transforms as tf
from torchvision.transforms.transforms import CenterCrop
from yacs.config import CfgNode
from default_paths import ROOT

from pathlib import Path


def load_augmentation_config(config_path: Path) -> CfgNode:
    """
    Loads augmentations configs defined as yaml files.
    """
    config = CfgNode()
    config.augmentations = CfgNode()
    config.augmentations.random_rotation = None
    config.augmentations.horizontal_flip = False
    config.augmentations.vertical_flip = False
    config.augmentations.normalize = None
    config.augmentations.resize = None
    config.augmentations.center_crop = None
    config.augmentations.expand_channels = False
    config.augmentations.random_crop = None
    config.augmentations.random_color_jitter = None

    yaml_config = config.clone()
    yaml_config.merge_from_file(config_path)
    return yaml_config


class ExpandChannels:
    """
    Transform 1-channel into 3-channel image, by copying the channel 3 times.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(data, 3, dim=0)


class Standardize:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.0] = 1.0
        return tf.functional.normalize(x, mean, std)


def default_augmentations_dataset(dataset_name):
    if dataset_name in ["entity13", "living17", "nonliving26", "entity30"]:
        dataset_name = "breeds"
    config = load_augmentation_config(ROOT / "classification" / "configs" / "augmentations" / f"{dataset_name}.yml")
    return get_augmentations_from_config(config)


def get_augmentations_from_config(config: CfgNode):
    """
    Return transformation pipeline as per config.
    Note: IF YOU LOAD A PRETRAINED IMAGENET MODEL THE TRANSFORMS WILL BE LOADED FROM THE MODEL WEIGHTS CONFIG
    AND ALL OTHER AUGMENTATIONS ARE IGNORE.
    (See https://pytorch.org/vision/stable/models.html)
    """
    transform_list, val_transforms = [], []
    if config.augmentations.random_crop is not None:
        transform_list.append(
            tf.RandomResizedCrop(config.augmentations.resize, scale=config.augmentations.random_crop)
        )
        val_transforms.append(tf.Resize(config.augmentations.resize))
    elif config.augmentations.resize is not None:
        transform_list.append(tf.Resize(config.augmentations.resize))
        val_transforms.append(tf.Resize(config.augmentations.resize))
    if config.augmentations.random_rotation is not None:
        transform_list.append(tf.RandomRotation(config.augmentations.random_rotation))
    if config.augmentations.horizontal_flip:
        transform_list.append(tf.RandomHorizontalFlip())
    if config.augmentations.vertical_flip:
        transform_list.append(tf.RandomVerticalFlip())
    if config.augmentations.random_color_jitter:
        transform_list.append(
            tf.ColorJitter(
                brightness=config.augmentations.random_color_jitter,
                contrast=config.augmentations.random_color_jitter,
                hue=0 if config.augmentations.expand_channels else config.augmentations.random_color_jitter,
                saturation=0 if config.augmentations.expand_channels else config.augmentations.random_color_jitter,
            )
        )
    if config.augmentations.center_crop is not None:
        transform_list.append(CenterCrop(config.augmentations.center_crop))
        val_transforms.append(CenterCrop(config.augmentations.center_crop))

    if config.augmentations.normalize is not None:
        transform_list.append(tf.Normalize(config.augmentations.normalize[0], config.augmentations.normalize[1]))
        val_transforms.append(tf.Normalize(config.augmentations.normalize[0], config.augmentations.normalize[1]))
    # To artificially transform 1-channel to 3-channels by copying the value over the 3 channels.
    if config.augmentations.expand_channels:
        transform_list.append(ExpandChannels())
        val_transforms.append(ExpandChannels())
    train_transforms = torch.nn.Sequential(*transform_list)
    return train_transforms, torch.nn.Sequential(*val_transforms)
