if __name__ == "__main__":
    """
    Script to launch training of all training configuration for a given dataset.
    """
    import argparse
    from classification.default_config import load_yaml_training_config
    from pathlib import Path
    from classification.train import train_model_main

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    args = parser.parse_args()
    config_dir = Path(__file__).parent / "configs" / "general"
    print(config_dir)
    for f in config_dir.glob("scratch/*.yml"):
        config = load_yaml_training_config(f, args.dataset)
        train_model_main(config)
    if args.dataset not in ["imagenet", "imageneta", "living17", "entity13", "nonliving26", "entity30"]:
        for f in config_dir.glob("pretrained/*.yml"):
            config = load_yaml_training_config(f, args.dataset)
            train_model_main(config)
