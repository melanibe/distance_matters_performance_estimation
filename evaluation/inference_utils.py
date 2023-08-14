import torch
import pickle
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
from classification.timm_wrap import TimmModelWrapper
from typing import Optional, Dict, Tuple
from torch.utils.data import DataLoader
import numpy as np

from data_handling.imagenet import IMAGENET_A_MASK

"""
Main file for inference functions and utils.
"""


def run_inference(
    dataloader: DataLoader,
    model: torch.nn.Module,
    temperature: Optional[float] = None,
    classwise_temperature: Optional[np.ndarray] = None,
    mask_outputs=None,
) -> Dict:
    """
    Main inference loop.
    Returns dictionary with logits, probas, targets, predictions,
    softmax_confidence with TS, without TS, with class-wise TS.
    If TS or classwise TS is not provided only returns raw softmax confidence.
    """
    if isinstance(model, TimmModelWrapper):
        model.model.eval().cuda()
        model.model_without_classifier.eval().cuda()
    else:
        model = model.eval().cuda()
    all_probas = []
    all_targets = []
    all_feats = []
    all_logits = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            if isinstance(batch, dict):
                data, target = batch["image"], batch["target"]
            else:
                data, target = batch[0], batch[1]
            if isinstance(model, TimmModelWrapper):
                # This is unefficient but the API is annoying to get pooled penultimate features
                feats = model.model_without_classifier(data.cuda())
                logits = model.model(data.cuda())
            else:
                data = model.preprocess(data.cuda())
                feats = model.get_features(data)
                logits = model.classify_features(feats)
            if mask_outputs is not None:
                logits = logits[:, mask_outputs]

            probas = torch.softmax(logits, 1)
            all_probas.append(probas.cpu())
            all_targets.append(target)
            all_feats.append(feats.cpu())
            all_logits.append(logits.cpu())
    if isinstance(model, TimmModelWrapper):
        model.model.eval().cpu()
        model.model_without_classifier.eval().cpu()
    else:
        model = model.cpu()

    results = {
        "probas": torch.cat(all_probas),
        "predictions": torch.argmax(torch.cat(all_probas), 1),
        "targets": torch.cat(all_targets),
        "feats": torch.cat(all_feats).numpy(),
        "softmax_confidence": torch.max(torch.cat(all_probas), 1)[0],
        "logits": torch.cat(all_logits),
    }
    if temperature is not None:
        results["logits_after_temperature"] = results["logits"] / temperature
        results["probas_after_temperature"] = torch.softmax(results["logits_after_temperature"], 1)
        results["softmax_confidence_after_temperature"] = torch.max(results["probas_after_temperature"], 1)[0]
    if classwise_temperature is not None and results["logits"].shape[0] > 1:
        results["logits_cs_ts"] = results["logits"] / classwise_temperature[results["predictions"]].reshape(-1, 1)
        assert results["logits_cs_ts"].shape == results["logits"].shape
        results["probas_cs_ts"] = torch.softmax(results["logits_cs_ts"], 1)
        results["softmax_confidence_cs_ts"] = torch.max(results["probas_cs_ts"], 1)[0]
    return results


def open_results_if_exists(filename: Path) -> Optional[Dict]:
    """
    Opens and loads pickle file to dict if exists.
    Else returns None.
    """
    if filename.exists():
        with open(filename, "rb") as handle:
            results = pickle.load(handle)
        return results
    print(f"Did not find {filename}")
    return None


def get_train_and_val_predictions(
    output_dir: Path, dataset: str, data_module: pl.LightningDataModule, model: torch.nn.Module
) -> Tuple[Dict, Dict]:
    """
    Get train and validation results dictionary for given model.

    Args:
        output_dir: Path to save model outputs
        dataset: str, name of dataset
        data_module: PL data module defining train and val loaders
        model: model to evaluate

    Returns:
        train_results, val_results
    """
    # Get calibration predictions
    train_results = open_results_if_exists(output_dir / "train.pickle")
    if not dataset.startswith("imagenet"):
        if train_results is None:
            train_results = run_inference(data_module.train_dataloader(), model)
            with open(output_dir / "train.pickle", "wb") as handle:
                pickle.dump(train_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    val_results = open_results_if_exists(output_dir / "val.pickle")
    if val_results is None:
        if dataset == "imageneta":
            mask = IMAGENET_A_MASK
        else:
            mask = None

        val_results = run_inference(data_module.val_dataloader(), model, mask_outputs=mask)
        with open(output_dir / "val.pickle", "wb") as handle:
            pickle.dump(val_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    all_class_in_val = val_results["probas"].shape[1] == torch.unique(val_results["targets"]).shape[0]
    all_class_predicted_in_val = val_results["probas"].shape[1] == torch.unique(val_results["predictions"]).shape[0]
    if not all_class_in_val:
        print(
            f"Found {torch.unique(val_results['targets']).shape[0]}"
            + f"classes in val out of {val_results['probas'].shape[1]} possible"
        )
    if not all_class_predicted_in_val:
        print(
            f"\n\nFound {torch.unique(val_results['predictions']).shape[0]}"
            + f"predicted classes in val out of {val_results['probas'].shape[1]} possible\n\n"
        )
    return train_results, val_results


def get_ood_predictions(
    eval_loader: DataLoader,
    name_eval_loader: str,
    model: torch.nn.Module,
    output_dir: Path,
    ts: float,
    cs_ts: np.ndarray,
) -> Dict:
    """
    Get predictions for a given dataloader, including temperature scaled confidences.

    Args:
        eval_loader: DataLoader to evaluate
        name_eval_loader: name of dataloader (to save results)
        model: model to evaluate
        output_dir: directory where to save results
        ts: temperature (global)
        cs_ts: array of temperature per class

    Returns:
        results: Dict
    """
    results = open_results_if_exists(output_dir / f"{name_eval_loader}.pickle")
    # results = None
    if results is None:
        if name_eval_loader == "imageneta":
            mask = IMAGENET_A_MASK
        else:
            mask = None

        results = run_inference(eval_loader, model, ts, cs_ts, mask_outputs=mask)
        with open(output_dir / f"{name_eval_loader}.pickle", "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results
