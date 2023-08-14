import torch
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from evaluation.inference_utils import open_results_if_exists
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict


class DistanceChecker:
    """
    Main class for DistanceChecker implementation.
    Implements fit method and methods to get kept samples both with
    global and class-wise distance checks.
    """

    def __init__(
        self, output_dir: Path, normalize_preds: bool = False, n_neighbors: bool = 25, quantile=0.990
    ) -> None:
        """
        Args:
            output_dir: Path to model directory, where to save distances
            normalize_preds: Whether to normalise predictions or not prior to fitting KNN.
            n_neighbors: Number of neighbours to use for KNN.
        """
        self.output_dir = output_dir
        self.normalize_preds = normalize_preds
        self.n_neighbors = n_neighbors
        self.quantile = quantile

    def fit(self, train_results: Optional[Dict], val_results: Dict) -> None:
        """
        Fit the Distance Checker. If train_results is provided use training results
        for fitting, else use validation results. Then get distance threshold from
        validation set.

        Args:
            train_results: Optional Dict with training results, as returned by run_inference function.
            val_results: Dict with validation results, as returned by run_inference function.
        """
        if train_results is not None:
            if self.normalize_preds:
                normalized_feats = train_results["feats"] / np.linalg.norm(
                    train_results["feats"], axis=1, keepdims=True
                )
            else:
                normalized_feats = train_results["feats"]
            if normalized_feats.shape[0] > 50000:
                selected_indices = np.random.choice(
                    np.arange(normalized_feats.shape[0]), min(normalized_feats.shape[0], 50000), replace=False
                )
                normalized_feats = normalized_feats[selected_indices]
            fit_on_train = True
        else:
            if self.normalize_preds:
                normalized_feats = val_results["feats"] / np.linalg.norm(val_results["feats"], axis=1, keepdims=True)
            else:
                normalized_feats = val_results["feats"]
            fit_on_train = False
        filename_calib = self.output_dir / f"distances_calib_n{self.n_neighbors}_norm{self.normalize_preds}.pickle"
        distances_calib = open_results_if_exists(filename_calib)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, n_jobs=6)
        self.nn.fit(normalized_feats)

        if distances_calib is None:
            if self.normalize_preds:
                distances_calib = self.nn.kneighbors(
                    val_results["feats"] / np.linalg.norm(val_results["feats"], axis=1, keepdims=True)
                )[0]
            else:
                distances_calib = self.nn.kneighbors(val_results["feats"])[0]
            # If the NN is fit on the same set as the calibration set used to compute the distance
            # threshold then we need to ignore the closest point (self).
            distances_calib = distances_calib[:, : self.n_neighbors] if fit_on_train else distances_calib[:, 1:]
            with open(filename_calib, "wb") as handle:
                pickle.dump(distances_calib, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.adt = np.quantile(distances_calib.mean(1), self.quantile)
        n_classes = val_results["logits"].shape[1]
        print(n_classes)
        self.adt_cs = np.ones(n_classes) * self.adt
        print(val_results["targets"].unique())
        # assert val_results["targets"].unique().sort() == torch.arange(n_classes)
        for target in range(n_classes):
            idx_class = torch.where((val_results["targets"] == target))[0]
            if idx_class.shape[0] >= 20:
                self.adt_cs[target] = np.quantile(distances_calib[idx_class].mean(1), self.quantile)
            # If there is less than 20 val samples from this class, use the global adt threshold.

    def get_kept_samples(self, ood_results: Dict, name_eval_loader: str) -> np.ndarray:
        """
        Get all kept sampled according to fitted distance checker using global distance threshold.

        Args:
            ood_results: Dict, as returned by run_inference
            name_eval_loader: str, name of data loader used for caching results to disk.

        Returns:
            kept_by_distance: np.ndarray[bool], boolean array of shape [n_samples,] indicating
                                whether each sample passes the distance check or not.
        """
        filename = f"distances_ood_{name_eval_loader}_n{self.n_neighbors}_norm{self.normalize_preds}.pickle"
        distances_ood = open_results_if_exists(self.output_dir / filename)
        if distances_ood is None:
            if self.normalize_preds:
                distances_ood = self.nn.kneighbors(
                    ood_results["feats"] / np.linalg.norm(ood_results["feats"], axis=1, keepdims=True),
                    n_neighbors=self.n_neighbors,
                )[0].mean(1)
            else:
                distances_ood = self.nn.kneighbors(ood_results["feats"], n_neighbors=self.n_neighbors)[0].mean(1)
            with open(self.output_dir / filename, "wb") as handle:
                pickle.dump(distances_ood, handle, protocol=pickle.HIGHEST_PROTOCOL)
        kept_by_distance = distances_ood < self.adt
        return kept_by_distance

    def get_kept_cs_samples(self, ood_results: Dict, name_eval_loader: str) -> np.ndarray:
        """
        Get all kept sampled according to fitted distance checker using class-wise distance threshold.

        Args:
            ood_results: Dict, as returned by run_inference
            name_eval_loader: str, name of data loader used for caching results to disk.

        Returns:
            kept_by_cs_distance: np.ndarray[bool], boolean array of shape [n_samples,] indicating
                                whether each sample passes the distance check or not.
        """
        filename = f"distances_ood_{name_eval_loader}_n{self.n_neighbors}_norm{self.normalize_preds}.pickle"
        distances_ood = open_results_if_exists(self.output_dir / filename)
        if distances_ood is None:
            if self.normalize_preds:
                distances_ood = self.nn.kneighbors(
                    ood_results["feats"] / np.linalg.norm(ood_results["feats"], axis=1, keepdims=True)
                )[0][:, : self.n_neighbors].mean(1)
            else:
                distances_ood = self.nn.kneighbors(ood_results["feats"])[0][:, : self.n_neighbors].mean(1)
            with open(self.output_dir / filename, "wb") as handle:
                pickle.dump(distances_ood, handle, protocol=pickle.HIGHEST_PROTOCOL)
        kept_by_cs_distance = distances_ood < self.adt_cs[ood_results["predictions"]]
        return kept_by_cs_distance


class MahaDistanceChecker:
    """
    DistanceChecker implementation using Mahalanobis distance instead of
    K-NN distance.
    Implements fit method and methods to get kept samples both with
    global and class-wise distance checks.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def fit(self, train_results: Dict, val_results: Dict) -> None:
        self.n_classes = val_results["logits"].shape[1]
        maha_train = open_results_if_exists(self.output_dir / "maha_train.pickle")
        if maha_train is None:
            if train_results is not None:
                tr_feats = train_results["feats"]
                tr_targets = train_results["targets"]
            else:
                tr_feats = val_results["feats"]
                tr_targets = val_results["targets"]
            if tr_feats.shape[0] > 50000:
                selected_indices = np.random.choice(
                    np.arange(tr_feats.shape[0]), min(tr_feats.shape[0], 50000), replace=False
                )
                tr_feats = tr_feats[selected_indices]
                tr_targets = tr_targets[selected_indices]

            mu = np.zeros((self.n_classes, tr_feats.shape[1]))
            sigma = np.zeros((self.n_classes, tr_feats.shape[1], tr_feats.shape[1]))
            for c in range(self.n_classes):
                idx_c = np.where(tr_targets == c)[0]
                mu[c] = np.mean(tr_feats[idx_c], axis=0)
                centered_feats = tr_feats[idx_c] - mu[c]  # b, f
                sigma[c] = np.transpose(centered_feats) @ centered_feats
            try:
                precision = np.linalg.inv(sigma.sum(0) / tr_targets.shape[0])
            except np.linalg.LinAlgError:
                precision = np.linalg.inv((sigma.sum(0) + np.eye(sigma.shape[1]) * 1e-12) / tr_targets.shape[0])
            maha_train = {"precision": precision, "mu": mu}
            with open(self.output_dir / "maha_train.pickle", "wb") as handle:
                pickle.dump(maha_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.mu = maha_train["mu"]
        self.precision = maha_train["precision"]

        maha_val = open_results_if_exists(self.output_dir / "maha_val.pickle")
        if maha_val is None:
            score_val_per_class = np.zeros((self.n_classes, val_results["feats"].shape[0]))
            for c in tqdm(range(self.n_classes)):
                score_val_per_class[c] = get_maha_distances(self.mu[c], self.precision, val_results["feats"])
            assert len(score_val_per_class) == self.n_classes
            scores = score_val_per_class.min(0)
            threshold = np.quantile(scores, 0.99)
            maha_val = {"scores": scores, "threshold": threshold}
            with open(self.output_dir / "maha_val.pickle", "wb") as handle:
                pickle.dump(maha_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.threshold = maha_val["threshold"]
        print("Finish init")

    def get_kept_samples(self, ood_results: Dict, name_eval_loader: str) -> np.ndarray:
        filename = f"maha_{name_eval_loader}.pickle"
        distances_ood = open_results_if_exists(self.output_dir / filename)
        if distances_ood is None:
            score_per_class = []
            for c in tqdm(range(self.n_classes)):
                score_per_class.append(get_maha_distances(self.mu[c], self.precision, ood_results["feats"]))
            assert len(score_per_class) == self.n_classes
            distances_ood = np.asarray(score_per_class).min(0)
            with open(self.output_dir / filename, "wb") as handle:
                pickle.dump(distances_ood, handle, protocol=pickle.HIGHEST_PROTOCOL)
        kept_by_distance = distances_ood < self.threshold
        return kept_by_distance


def get_maha_distances(mu, precision, feats):
    feats = feats - mu
    return ((feats @ precision) * feats).sum(1)
