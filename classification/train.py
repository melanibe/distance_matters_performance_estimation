from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from classification.default_config import load_yaml_training_config
from classification.load_model_and_config import get_modules, get_output_dir_from_config


def train_model_main(config, raise_if_exists=False):
    if isinstance(config.seed, int):
        config.seed = [config.seed]
    all_seeds = config.seed

    for seed in all_seeds:
        config.seed = seed
        output_dir = get_output_dir_from_config(config)
        if (output_dir / "best.ckpt").exists():
            if raise_if_exists:
                raise FileExistsError("The best checkpoint already exists in this folder")
            else:
                print("The best checkpoint already exists in this folder. Not running training.")
            continue

        pl.seed_everything(config.seed, workers=True)

        data_module, model_module = get_modules(config)

        checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
        checkpoint_callback_best = ModelCheckpoint(
            dirpath=output_dir, monitor=config.training.metric_to_monitor, mode="max", filename="best"
        )
        lr_monitor = LearningRateMonitor()
        early_stopping = EarlyStopping(
            monitor=config.training.metric_to_monitor,
            mode=config.training.metric_to_monitor_mode,
            patience=round(1.5 * config.training.patience_for_scheduler),
        )

        logger = TensorBoardLogger(output_dir, name="tensorboard")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=config.training.num_epochs,
            logger=logger,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_best,
                lr_monitor,
                early_stopping,
            ],
            val_check_interval=None if config.dataset != "wilds_icam" else 2000,
        )
        trainer.fit(model_module, data_module)


if __name__ == "__main__":
    """
    Script to run one particular configuration.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    args = parser.parse_args()

    config = load_yaml_training_config(Path(__file__).parent / "configs" / args.config, args.dataset)
    train_model_main(config)
