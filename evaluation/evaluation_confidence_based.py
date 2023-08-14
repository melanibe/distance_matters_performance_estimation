import pickle
import torch
from classification.load_model_and_config import get_config_data_model_for_eval
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tabulate import tabulate
from evaluation.confidence_estimates import ConfidenceBasedAccuracyEstimator
from evaluation.distance_checker import DistanceChecker, MahaDistanceChecker
import ot
from evaluation.inference_utils import (
    get_train_and_val_predictions,
    open_results_if_exists,
    get_ood_predictions,
)

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import time
from yacs.config import CfgNode
from typing import Union


def run_evaluation(config_name_or_path: Union[CfgNode, str], dataset: str) -> None:
    """
    Main evaluation loop for a given model configuration, dataset configuration.
    Computes all estimates and error metrics and save as metrics.csv to model output directory.

    Args:
        config_name_or_path: load config object or relative path to config for model configuration
                            to load.
        dataset: name of dataset to evaluate.
    """

    # Load all models for that training configuration (usually 3 seeds)
    try:
        config, data_modules, models, output_dirs = get_config_data_model_for_eval(config_name_or_path, dataset)
    # this is for skipping the broken models in timm
    except Exception as inst:
        print(type(inst))
        print(inst)
        return

    # Too slow in ImageNet - skip Maha
    get_maha = not dataset.startswith("imagenet")

    get_cot = True

    config.dataset = dataset
    metrics = pd.DataFrame()

    # If no trained models are found, return
    if len(models) == 0:
        return

    # Main evaluation loop, iterate over all model instances
    for model, output_dir, data_module in zip(models, output_dirs, data_modules):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # If metrics csv is already present, skip computation
        if (output_dir.parent / "metrics.csv").exists():
            df = pd.read_csv(output_dir.parent / "metrics.csv")
            if "predicted_cot_ts" in df.columns:
                return pd.read_csv(output_dir.parent / "metrics.csv")

        # Get predictions
        train_results, val_results = get_train_and_val_predictions(output_dir, dataset, data_module, model)

        if dataset in ["imageneta"]:
            one_hot_val = OneHotEncoder(categories=np.arange(200).reshape(1, -1), sparse=False).fit_transform(
                val_results["targets"].reshape(-1, 1)
            )
            print("here")
        else:
            one_hot_val = OneHotEncoder(
                categories=np.arange(data_module.num_classes).reshape(1, -1), sparse=False
            ).fit_transform(val_results["targets"].reshape(-1, 1))

        # Fit TS, ATC, DOC
        accuracy_estimator = ConfidenceBasedAccuracyEstimator()
        accuracy_estimator.fit(val_results)
        if accuracy_estimator.accuracy_calibration == 0.0:
            continue

        # Fit ADT
        distance_checker = DistanceChecker(output_dir, normalize_preds=False)
        distance_checker.fit(train_results, val_results)

        # Fit ADT-Maha
        if get_maha:
            maha_distance_checker = MahaDistanceChecker(output_dir)
            maha_distance_checker.fit(train_results, val_results)

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
            start_time = time.time()
            kept_by_distance = distance_checker.get_kept_samples(ood_results, name_eval_loader)
            print(f"It took {time.time() - start_time} to get the kept samples.")
            kept_by_cs_distance = distance_checker.get_kept_cs_samples(ood_results, name_eval_loader)

            # Compute all the metrucs
            current_metrics_dict = {
                "dataset": name_eval_loader,
                "seed": output_dir.stem,
            }

            current_metrics_dict["accuracy"] = accuracy_score(ood_results["targets"], ood_results["predictions"])

            # ATC estimates
            current_metrics_dict["predicted_atc"] = accuracy_estimator.get_atc_estimate(ood_results)
            current_metrics_dict["predicted_atc_ts"] = accuracy_estimator.get_atc_ts_estimate(ood_results)
            current_metrics_dict["predicted_atc_cs_ts"] = accuracy_estimator.get_atc_cs_ts_estimate(ood_results)

            # DOC estimates
            current_metrics_dict["predicted_doc"] = accuracy_estimator.get_doc_estimate(ood_results)
            current_metrics_dict["predicted_doc_ts"] = accuracy_estimator.get_doc_ts_estimate(ood_results)
            current_metrics_dict["predicted_doc_cs_ts"] = accuracy_estimator.get_doc_cs_ts_estimate(ood_results)

            # ATC - DIST estimates
            current_metrics_dict["predicted_atc_dist"] = accuracy_estimator.get_atc_dist_estimate(
                ood_results, kept_by_distance
            )
            current_metrics_dict["predicted_dist_only"] = kept_by_distance.mean()
            current_metrics_dict["predicted_atc_ts_dist"] = accuracy_estimator.get_atc_ts_dist_estimate(
                ood_results, kept_by_distance
            )
            current_metrics_dict["predicted_atc_cs_ts_dist"] = accuracy_estimator.get_atc_cs_ts_dist_estimate(
                ood_results, kept_by_distance
            )
            current_metrics_dict["predicted_atc_ts_csdist"] = accuracy_estimator.get_atc_ts_dist_estimate(
                ood_results, kept_by_cs_distance
            )
            current_metrics_dict["predicted_atc_cs_ts_csdist"] = accuracy_estimator.get_atc_cs_ts_dist_estimate(
                ood_results, kept_by_cs_distance
            )

            # MAHA - ATC - DIST estimates
            if get_maha:
                maha_kept_by_distance = maha_distance_checker.get_kept_samples(ood_results, name_eval_loader)
                current_metrics_dict["predicted_atc_ts_maha_dist"] = accuracy_estimator.get_atc_ts_dist_estimate(
                    ood_results, maha_kept_by_distance
                )

                current_metrics_dict["predicted_atc_cs_ts_maha_dist"] = accuracy_estimator.get_atc_cs_ts_dist_estimate(
                    ood_results, maha_kept_by_distance
                )

            # COT estimates with global TS and classwise TS
            if get_cot:
                cot_estimate_dict = open_results_if_exists(output_dir / f"cot_estimates_{name_eval_loader}.pickle")
                if cot_estimate_dict is None:
                    all_cot_estimates_ts = []
                    all_cot_estimates_cs_ts = []

                    # Get randomised indexes
                    all_indexes = torch.randperm(ood_results["softmax_confidence_after_temperature"].shape[0])

                    # Split into batches if big
                    if all_indexes.shape[0] > 5000:
                        batched_iterable = torch.split(all_indexes, 2500)
                        # do not use more than 25k samples for estimation otherwise it takes 90mins per estimation
                        if len(batched_iterable) > 10:
                            batched_iterable = batched_iterable[:10]
                    else:
                        batched_iterable = [all_indexes]

                    start_time = time.time()

                    # Iterate over batches
                    for batch in tqdm(batched_iterable):
                        if batch.shape[0] < 1000:
                            continue  # ignore the estimate if the split is too small to give reliable estimates.
                        M = ot.dist(
                            one_hot_val,
                            ood_results["probas_after_temperature"][batch].numpy(),
                            metric="minkowski",
                            p=1,
                        )
                        dist = 0.5 * ot.emd2([], [], M, numItermax=500000)
                        all_cot_estimates_ts.append(dist)
                        M = ot.dist(one_hot_val, ood_results["probas_cs_ts"][batch].numpy(), metric="minkowski", p=1)
                        all_cot_estimates_cs_ts.append(0.5 * ot.emd2([], [], M, numItermax=500000))

                    cot_estimate_dict = {"ts": all_cot_estimates_ts, "cs_ts": all_cot_estimates_cs_ts}
                    print(f"It took {(time.time() - start_time) / 2} to get one COT estimate.")

                    # Save for future evaluation passes.
                    with open(output_dir / f"cot_estimates_{name_eval_loader}.pickle", "wb") as handle:
                        pickle.dump(cot_estimate_dict, handle)

                current_metrics_dict["predicted_cot_cs_ts"] = 1 - np.asarray(cot_estimate_dict["cs_ts"]).mean()
                current_metrics_dict["predicted_cot_ts"] = 1 - np.asarray(cot_estimate_dict["ts"]).mean()

            # Convert dict to dataframe and add absolute errors columns
            current_metrics = pd.DataFrame(current_metrics_dict, index=[0])
            for c in current_metrics.columns:
                if c.startswith("predicted"):
                    current_metrics[f"error_{c[10:]}"] = current_metrics[c].apply(
                        lambda x: np.abs(x - current_metrics_dict["accuracy"]) if x is not None else np.nan
                    )

            # Append to current to metrics df
            metrics = pd.concat([metrics, current_metrics], ignore_index=True)

            # Print during execution
            error_cols = [i for i in current_metrics.columns if i.startswith("error")]
            print(
                tabulate(
                    current_metrics[["dataset", "accuracy"] + error_cols],
                    headers="keys",
                )
            )

    # Save results to disk and print
    if len(metrics) != 0:
        error_cols = [i for i in metrics.columns if i.startswith("error")]
        aggregated_metrics = metrics[error_cols].aggregate(func=["mean", "std"])

        print(tabulate(aggregated_metrics, headers="keys"))

        aggregated_metrics.to_csv(output_dir.parent / "aggregated.csv")
        metrics.to_csv(output_dir.parent / "metrics.csv")


if __name__ == "__main__":
    """
    Script to evaluate all available trained model configuration for a given dataset.
    Assumes trained models are placed in ROOT/outputs/[TEST_DATASET]/[MODEL_NAME]/[RUN_NAME]/SEED_[S]

    Usage: python evaluation/run_evaluation.py --dataset [TEST_DATASET]
    """
    import argparse
    from pathlib import Path
    from classification.default_config import load_yaml_training_config
    import timm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # If it is not ImageNet use our own models
    if not args.dataset.startswith("imagenet"):
        config_dir = Path(__file__).parent.parent / "classification" / "configs" / "general"
        for config_file in config_dir.glob("scratch/*.yml"):
            print(config_file)
            run_evaluation(config_file, args.dataset)
        # Don't use the pretrained configuration if the model is from BREEDS (subset of ImageNet)
        if args.dataset not in ["entity13", "living17", "nonliving26", "entity30"]:
            for config_file in config_dir.glob("pretrained/*.yml"):
                print(config_file)
                run_evaluation(config_file, args.dataset)

    # Else use the models trained from the timm package
    else:
        default_config = load_yaml_training_config(None, "imagenet")
        default_config.dataset = args.dataset
        for model_name in timm.list_models(pretrained=True):
            if any(s in model_name for s in ["resn", "efficientnet", "densen", "darknet", "convnext", "convmixer"]):
                if model_name.endswith("_in21k") or model_name.endswith("_in22k"):
                    continue
                print(model_name)
                print(f"Evaluating {model_name}")
                default_config.evaluation.timm_model_to_evaluate = model_name
                run_evaluation(default_config, args.dataset)
            else:
                continue
