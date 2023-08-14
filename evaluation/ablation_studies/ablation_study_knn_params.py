from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tabulate import tabulate

from classification.load_model_and_config import get_config_data_model_for_eval
from evaluation.confidence_estimates import ConfidenceBasedAccuracyEstimator
from evaluation.distance_checker import DistanceChecker
from evaluation.inference_utils import get_ood_predictions, get_train_and_val_predictions
from yacs.config import CfgNode
from typing import Union


def run_ablation(config_name_or_path: Union[CfgNode, str], dataset: str) -> pd.DataFrame:
    """
    Runs ablation on number of neighbours [5, 25, 50] and features normalisation for distance checker
    for particular dataset and model config.

    Returns:
        metrics: pd.DataFrame with metrics for that particular model, dataset combination.
    """
    try:
        config, data_modules, models, output_dirs = get_config_data_model_for_eval(config_name_or_path, dataset)
    except RuntimeError:
        # this is for skipping the broken models in timm
        return
    config.dataset = dataset
    metrics = pd.DataFrame()

    if len(models) == 0:
        return

    for model, output_dir, data_module in zip(models, output_dirs, data_modules):
        output_dir = Path(output_dir)
        seed = output_dir.stem
        save_dir = output_dir.parent / "ablation"
        save_dir.mkdir(parents=True, exist_ok=True)

        if (save_dir / "metrics.csv").exists():
            df = pd.read_csv(save_dir / "metrics.csv")
            if "error_atc_ts_dist_truetta_rot" in df.columns:
                print(f"{save_dir} / metrics.csv exists")
                return

        # Get predictions
        train_results, val_results = get_train_and_val_predictions(output_dir, dataset, data_module, model)

        # Fit TS, ATC, DOC
        accuracy_estimator = ConfidenceBasedAccuracyEstimator()
        accuracy_estimator.fit(val_results)
        if accuracy_estimator.accuracy_calibration == 0.0:
            continue

        # Fit ADT
        for neighbours in [25, 5, 50]:
            for normalise in [True, False]:
                adt_estimator = DistanceChecker(output_dir, n_neighbors=neighbours, normalize_preds=normalise)
                adt_estimator.fit(train_results, val_results)

                for (name_eval_loader, eval_loader) in data_module.get_all_ood_dataloaders():
                    print(f"Processing {name_eval_loader}")

                    ood_results = get_ood_predictions(
                        eval_loader,
                        name_eval_loader,
                        model,
                        output_dir,
                        accuracy_estimator.ts,
                        accuracy_estimator.cs_ts,
                    )

                    # Get DIST-estimate
                    kept_by_distance = adt_estimator.get_kept_samples(ood_results, name_eval_loader)
                    kept_by_cs_distance = adt_estimator.get_kept_cs_samples(ood_results, name_eval_loader)

                    # Compute all the metrics
                    current_metrics_dict = {"dataset": name_eval_loader, "seed": seed}

                    current_metrics_dict["seed"] = seed
                    current_metrics_dict["n_neighbors"] = neighbours
                    current_metrics_dict["normalise"] = normalise

                    current_metrics_dict["accuracy"] = accuracy_score(
                        ood_results["targets"], ood_results["predictions"]
                    )
                    print(current_metrics_dict["accuracy"])

                    # ATC estimates
                    current_metrics_dict["predicted_atc_ts"] = accuracy_estimator.get_atc_ts_estimate(ood_results)
                    current_metrics_dict["predicted_atc_cs_ts"] = accuracy_estimator.get_atc_cs_ts_estimate(
                        ood_results
                    )

                    # ATC - DIST estimates
                    current_metrics_dict["predicted_atc_ts_dist"] = accuracy_estimator.get_atc_ts_dist_estimate(
                        ood_results, kept_by_distance
                    )
                    current_metrics_dict[
                        "predicted_atc_cs_ts_csdist"
                    ] = accuracy_estimator.get_atc_cs_ts_dist_estimate(ood_results, kept_by_cs_distance)

                    current_metrics = pd.DataFrame(current_metrics_dict, index=[0])
                    for c in current_metrics.columns:
                        if c.startswith("predicted"):
                            current_metrics[f"error_{c[10:]}"] = current_metrics[c].apply(
                                lambda x: np.abs(x - current_metrics_dict["accuracy"]) if x is not None else np.nan
                            )
                    metrics = pd.concat([metrics, current_metrics], ignore_index=True)
                    error_cols = [i for i in current_metrics.columns if i.startswith("error")]
                    print(
                        tabulate(
                            current_metrics[["dataset", "accuracy", "n_neighbors", "normalise"] + error_cols],
                            headers="keys",
                        )
                    )

    if len(metrics) != 0:
        metrics.to_csv(save_dir / "metrics.csv")
    return metrics


if __name__ == "__main__":
    """
    This is the main experiment script for our second ablation study on
    number of neighbours and features normalisation.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    all_metrics = pd.DataFrame()

    config_dir = Path(__file__).parent.parent / "classification" / "configs" / "general"
    for config_file in config_dir.glob("scratch/*.yml"):
        print(config_file)
        metrics = run_ablation(config_file, args.dataset)
        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
    if args.dataset not in ["entity13", "living17", "nonliving26"]:
        for config_file in config_dir.glob("pretrained/*.yml"):
            print(config_file)
            metrics = run_ablation(config_file, args.dataset)
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

    error_cols = [i for i in metrics.columns if i.startswith("error")]
    aggregated_metrics = (
        all_metrics[error_cols + ["n_neighbors", "normalise"]]
        .groupby(by=["n_neighbors", "normalise"])
        .aggregate(func=["mean", "std"])
    )
    print(tabulate(aggregated_metrics, headers="keys"))
    all_metrics.to_csv(f"outputs/{args.dataset}/ablation_all.csv")
    aggregated_metrics.to_csv(f"outputs/{args.dataset}/ablation_aggregated.csv")
