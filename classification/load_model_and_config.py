import os
from pathlib import Path
from typing import List

import torchvision
from pytorch_lightning.utilities.seed import seed_everything
from yacs.config import CfgNode

from classification.classification_module import ClassificationModule
from classification.default_config import load_yaml_training_config
from classification.timm_wrap import TimmModelWrapper
from data_handling.augmentations import default_augmentations_dataset
from data_handling.breeds import Entity13DataModule, Living17DataModule, NonLiving26DataModule, Entity30DataModule
from data_handling.cifar import CIFAR10DataModule
from data_handling.imagenet import ImageNetDataModule, ImageNetADataModule
from data_handling.medmnist import PathMNISTModule
from data_handling.wilds_module import WILDSCameLyon17, WILDSiCam, WILDSrr1, WILDSFMoW
from data_handling.mnist import MNISTDataModule
from data_handling.pacs import PACSModule
from default_paths import ROOT


def get_modules(config, shuffle_training: bool = True):

    _dataset_name_to_module_cls = {
        "cifar10": CIFAR10DataModule,
        "wilds_camelyon": WILDSCameLyon17,
        "wilds_icam": WILDSiCam,
        "pathmnist": PathMNISTModule,
        "imagenet": ImageNetDataModule,
        "imageneta": ImageNetADataModule,
        "living17": Living17DataModule,
        "entity13": Entity13DataModule,
        "nonliving26": NonLiving26DataModule,
        "wilds_rr1": WILDSrr1,
        "entity30": Entity30DataModule,
        "wilds_fmow": WILDSFMoW,
        "mnist": MNISTDataModule,
        "pacs": PACSModule,
    }

    data_module_cls = _dataset_name_to_module_cls[config.dataset]
    data_module = data_module_cls(
        num_workers=min(16, os.cpu_count()),
        shuffle=shuffle_training,
    )

    if config.evaluation.timm_model_to_evaluate is not None:
        module = TimmModelWrapper(config.evaluation.timm_model_to_evaluate)
        data_module.preprocess = module.transform
        config.model_name = module.model_name

    # Create PL module - unless this is a predefined model loaded from timm
    else:
        train_transforms_module, val_transforms_module = default_augmentations_dataset(config.dataset)
        module = ClassificationModule(
            train_transform_module=train_transforms_module,
            val_transform_module=val_transforms_module,
            encoder_name=config.model_name,
            num_classes=data_module.num_classes,
            lr=config.training.lr,
            patience_for_scheduler=config.training.patience_for_scheduler,
            metric_to_monitor=config.training.metric_to_monitor,
            metric_to_monitor_mode=config.training.metric_to_monitor_mode,
            weight_decay=config.training.weight_decay,
        )

    return data_module, module


def get_output_dir_from_config(config: CfgNode) -> Path:
    return ROOT / "outputs" / config.dataset / config.model_name / f"{config.run_name}" / f"seed_{config.seed}"


def get_torchvision_weight_object_from_string(weightobj_as_str):
    weight_class_name, weight_name = weightobj_as_str.split(".")
    w_cls = getattr(torchvision.models, weight_class_name)
    return getattr(w_cls, weight_name)


def load_model_from_checkpoint(config, model_module):
    # Nothing to do timm module are always loaded as pretrained models.
    if config.evaluation.timm_model_to_evaluate is not None:
        model_module.model.eval()
        return model_module
    if config.evaluation.torchvision_model_to_evaluate is not None:
        model = model_module.model
        # Loads torchvision models weights as per API of v0.14.0
        # e.g. ResNet50_Weights.IMAGENET1K_V1
        torchvision_weights = get_torchvision_weight_object_from_string(
            config.evaluation.torchvision_model_to_evaluate
        )
        try:
            model.net.load_state_dict(torchvision_weights.get_state_dict(progress=True))
        except RuntimeError:
            _load_state_dict(model.net, torchvision_weights)
        return model.eval()
    return model_module.load_from_checkpoint(
        Path(config.output_dir) / "best.ckpt",
        encoder_name=config.model_name,
        train_transform_module=model_module.train_transform_module,
        val_transform_module=model_module.model.preprocess,
    ).model.eval()


def _load_state_dict(model, weights, progress: bool = True) -> None:
    import re

    # from torchvision.models.densenet
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def get_config_data_model_for_eval(config_or_config_name: str, dataset):
    if isinstance(config_or_config_name, CfgNode):
        config = config_or_config_name
    elif isinstance(config_or_config_name, Path):
        config = load_yaml_training_config(config_or_config_name, dataset)
    else:
        config = load_yaml_training_config(Path(__file__).parent / config_or_config_name, dataset)
    models = []
    output_dirs = []
    data_modules = []
    all_seeds = config.seed if isinstance(config.seed, List) else [config.seed]
    seed_everything(all_seeds[0], workers=True)
    for seed in all_seeds:
        try:
            config.seed = seed
            data_module, model_module = get_modules(config, shuffle_training=False)
            data_module.setup()
            config.output_dir = str(get_output_dir_from_config(config))
            models.append(load_model_from_checkpoint(config, model_module))
            output_dirs.append(config.output_dir)
            data_modules.append(data_module)
        except FileNotFoundError:
            continue
    config.seed = all_seeds
    return config, data_modules, models, output_dirs
