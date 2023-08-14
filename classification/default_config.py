from typing import List, Optional
from yacs.config import CfgNode
from pathlib import Path

config = CfgNode()

config.seed = [11]
config.run_name = "run"
config.model_name = None

config.training = CfgNode()
config.training.lr = 1e-3
config.training.num_epochs = 200
config.training.batch_size = 128
config.training.patience_for_scheduler = 25
config.training.metric_to_monitor = "Val/AUROC"
config.training.metric_to_monitor_mode = "max"
config.training.val_check_interval = None
config.training.weight_decay = 0.0
config.training.use_training_augmentations = True

config.evaluation = CfgNode()
config.evaluation.torchvision_model_to_evaluate = None
config.evaluation.timm_model_to_evaluate = None


def load_yaml_training_config(config_path: Optional[Path], dataset: str) -> CfgNode:
    """
    Loads augmentations configs defined as yaml files.
    """
    yaml_config = config.clone()
    if config_path is not None:
        yaml_config.merge_from_file(config_path)
    yaml_config.dataset = dataset
    validate_config(yaml_config)
    return yaml_config


def validate_config(config):
    if not config.dataset.startswith("imagenet"):
        if config.evaluation.torchvision_model_to_evaluate is not None:
            raise ("You are loading weights from an ImageNet model but your dataset is not ImageNet")
        if config.evaluation.timm_model_to_evaluate is not None:
            raise ("You are loading weights from an ImageNet model but your dataset is not ImageNet")

    else:
        if (
            config.evaluation.torchvision_model_to_evaluate is not None
            and config.evaluation.timm_model_to_evaluate is not None
        ):
            raise ValueError("You can not load a torchvision and a timm model in the same config. Pls fix.")
        if (
            config.evaluation.torchvision_model_to_evaluate is not None
            or config.evaluation.timm_model_to_evaluate is not None
        ) and isinstance(config.seed, List):
            raise ValueError(
                """
                You specified several seeds in the config for evaluation of one specific model.
                This does not make sense as seed has no effect.
                """
            )
