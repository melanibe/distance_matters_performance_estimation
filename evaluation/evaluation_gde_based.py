from classification.load_model_and_config import get_config_data_model_for_eval
import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from evaluation.confidence_estimates import ConfidenceBasedAccuracyEstimator
from evaluation.distance_checker import DistanceChecker

from evaluation.inference_utils import (
    get_train_and_val_predictions,
    get_ood_predictions,
)

from collections import defaultdict
from itertools import combinations, permutations
from scipy.stats import wilcoxon
from yacs.config import CfgNode
from typing import Union


def run_evaluation_agreement(config_name_or_path: Union[CfgNode, str], dataset: str) -> pd.DataFrame:
    """
    Run evaluation of agreement based accuracy estimation (GDE) for a given training configuration.
    Assumes there is a least 2 models trained with this configuration available (two different seeds).
    """
    config, data_modules, models, output_dirs = get_config_data_model_for_eval(config_name_or_path, dataset)
    config.dataset = dataset
    metrics = pd.DataFrame()

    # Need to have a least two trained models for agreement
    # based accuracy estimation
    if len(models) <= 1:
        return

    ood_results = defaultdict(list)
    kept_by_distance = defaultdict(list)
    kept_by_cs_distance = defaultdict(list)

    for model, output_dir, data_module in zip(models, output_dirs, data_modules):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(output_dir)

        train_results, val_results = get_train_and_val_predictions(output_dir, dataset, data_module, model)

        # Fit Distance Checker
        distance_checker = DistanceChecker(output_dir)
        distance_checker.fit(train_results, val_results)

        # Fit TS, ATC, DOC
        accuracy_estimator = ConfidenceBasedAccuracyEstimator()
        accuracy_estimator.fit(val_results)

        for (name_eval_loader, eval_loader) in data_module.get_all_ood_dataloaders():
            ood_results[name_eval_loader].append(
                get_ood_predictions(
                    eval_loader,
                    name_eval_loader,
                    model,
                    output_dir,
                    accuracy_estimator.ts,
                    accuracy_estimator.cs_ts,
                )
            )

            # Get DIST-estimate
            kept_by_distance[name_eval_loader].append(
                distance_checker.get_kept_samples(ood_results[name_eval_loader][-1], name_eval_loader)
            )
            if distance_checker.adt_cs is not None:
                kept_by_cs_distance[name_eval_loader].append(
                    distance_checker.get_kept_cs_samples(ood_results[name_eval_loader][-1], name_eval_loader)
                )

    for name_eval_loader in ood_results.keys():
        for i, j in combinations(np.arange(len(ood_results[name_eval_loader])), 2):
            for ref, aux in permutations([i, j]):
                current_metrics_dict = {
                    "dataset": name_eval_loader,
                    "ref": config.seed[ref],
                    "aux": config.seed[aux],
                }
                current_metrics_dict["accuracy"] = (
                    (
                        ood_results[name_eval_loader][ref]["predictions"]
                        == ood_results[name_eval_loader][ref]["targets"]
                    )
                    .float()
                    .mean()
                    .item()
                )
                current_metrics_dict["predicted_aggreement"] = (
                    (
                        ood_results[name_eval_loader][ref]["predictions"]
                        == ood_results[name_eval_loader][aux]["predictions"]
                    )
                    .float()
                    .mean()
                    .item()
                )
                current_metrics_dict["predicted_aggreement_w_dist"] = (
                    (
                        (
                            ood_results[name_eval_loader][ref]["predictions"]
                            == ood_results[name_eval_loader][aux]["predictions"]
                        )
                        & kept_by_distance[name_eval_loader][ref]
                    )
                    .float()
                    .mean()
                    .item()
                )
                if distance_checker.adt_cs is not None:
                    current_metrics_dict["predicted_aggreement_w_csdist"] = (
                        (
                            (
                                ood_results[name_eval_loader][ref]["predictions"]
                                == ood_results[name_eval_loader][aux]["predictions"]
                            )
                            & kept_by_cs_distance[name_eval_loader][ref]
                        )
                        .float()
                        .mean()
                        .item()
                    )

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
            metrics[["dataset"] + error_cols]
            .groupby("dataset")
            .aggregate(func=lambda x: np.nanmean(x * 100))
            .dropna(),
            headers="keys",
        )
    )

    metrics.to_csv(output_dir.parent / "metrics_agreement.csv")

    return metrics


def is_significant(ref, new):
    if (ref - new).sum() == 0:
        return ""
    p = wilcoxon(ref, new, nan_policy="omit", alternative="two-sided")[1]
    if p < 1e-3:
        return f" **({p:.0E})"
    if p < 0.05:
        return f" *({p:.0E})"
    return f" ({p:.3f})"


if __name__ == "__main__":
    """
    Main script to run the analysis on GDE versus GDE+DistCS.
    Usage:
    python evaluation/run_evaluation_agreement_based.py --dataset [TEST_DATASET]
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

    if args.dataset != "imagenet":
        config_dir = Path(__file__).parent.parent / "classification" / "configs" / "general"
        for config_file in config_dir.glob("scratch/*.yml"):
            print(config_file)
            metrics = run_evaluation_agreement(config_file, args.dataset)
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
        if args.dataset not in ["entity13", "living17", "nonliving26"]:
            for config_file in config_dir.glob("pretrained/*.yml"):
                print(config_file)
                metrics = run_evaluation_agreement(config_file, args.dataset)
                all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
    else:
        raise ValueError(
            "Can't run this analysis with ImageNet" + "we only have one seed for each trained model in timm."
        )

    error_cols = ["error_aggreement", "error_aggreement_w_csdist"]

    results = all_metrics[error_cols].aggregate(
        func=lambda x: f"{np.nanmean(x * 100, keepdims=True)[0]:.2f}"
        + is_significant(all_metrics["error_aggreement"].values, x)
    )

    results.to_csv(f"/data/performance_estimation/outputs/{args.dataset}/aggrement_summary.csv")

    print(tabulate(results))
