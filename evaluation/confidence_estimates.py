import torch
from sklearn.metrics import accuracy_score
import numpy as np
import scipy
from typing import Tuple, Dict


def learn_temperature_atc_and_doc(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    """
    Learns temperature and associated ATC and DOC for a given set of
    logits and labels.

    Args:
        logits: torch.tensor [n_samples, n_classes]
        labels: torch.tensor [n_samples]
    """

    # Optimise TS
    def temp_opt_func(t):
        ts_logits = logits / t
        return torch.nn.CrossEntropyLoss()(ts_logits, labels)

    temperature = scipy.optimize.minimize(fun=temp_opt_func, x0=np.array([1.0]), method="Nelder-Mead", tol=1e-07).x[0]

    accuracy = (torch.argmax(torch.softmax(logits, 1), 1) == labels).float().mean()
    softmax_confidence_ts = torch.max(torch.softmax(logits / temperature, 1), 1)[0]

    # Optimise atc
    atc_ts = scipy.optimize.minimize(
        fun=lambda x: np.abs(np.mean(softmax_confidence_ts.numpy() > x) - accuracy),
        x0=1.0,
        method="Nelder-Mead",
        tol=1e-07,
    ).x[0]

    # Optimise doc
    doc_ts = softmax_confidence_ts.mean() - accuracy

    return temperature, atc_ts, doc_ts


class ConfidenceBasedAccuracyEstimator:
    """
    Class to build ATC, DoC and classwise versions estimators.
    To use, call fit on the ID validation dataset first.
    Use one of the other methods to create the estimator you are looking for.

    Inference results and validation results are expected as Dict in the structure
    returned by the run_inference function.
    """

    def fit(self, val_results: Dict) -> None:
        """
        Learns temperature, ATC, DoC and their class-wise versions from validation
        results.

        val_results: dict, required keys 'targets', 'predictions', 'logits', 'softmax_confidence' (max softmax)
        """
        self.accuracy_calibration = accuracy_score(val_results["targets"], val_results["predictions"])
        print(self.accuracy_calibration)

        # For broken models in timm
        if self.accuracy_calibration == 0.0:
            return

        # Get ATC threshold
        self.atc = scipy.optimize.minimize(
            fun=lambda x: np.abs(np.mean(val_results["softmax_confidence"].numpy() > x) - self.accuracy_calibration),
            x0=1.0,
            method="Nelder-Mead",
            tol=1e-07,
        ).x[0]
        self.doc = val_results["softmax_confidence"].mean() - self.accuracy_calibration

        # Get overall TS and corresponding ATC
        self.ts, self.atc_ts, self.doc_ts = learn_temperature_atc_and_doc(
            val_results["logits"], val_results["targets"]
        )

        # Get classwise TS and corresponding ATC
        n_classes = val_results["logits"].shape[1]
        predictions = torch.argmax(torch.softmax(val_results["logits"], 1), 1)
        self.cs_ts = torch.ones(n_classes) * self.ts
        self.atc_cs_ts = torch.ones(n_classes) * self.atc_ts
        self.doc_cs_ts = torch.ones(n_classes) * self.doc_ts
        for target in range(n_classes):
            idx_class = torch.where(predictions == target)[0]
            if idx_class.shape[0] < 20:
                print(
                    f"\n\nFound less than N=20 predicted class {target} using global T, ATC, DOC for that class.\n\n"
                )
                continue
            logits_class = val_results["logits"][idx_class]
            labels_class = val_results["targets"][idx_class]
            ts_class, atc_class, doc_class = learn_temperature_atc_and_doc(logits_class, labels_class)
            self.cs_ts[target] = ts_class
            self.atc_cs_ts[target] = atc_class
            self.doc_cs_ts[target] = doc_class

    def get_atc_estimate(self, inference_results: Dict) -> float:
        """
        Get ATC estimate without temperature scaling
        Args:
            inference_results: dict with 'softmax_confidence' key
        Returns:
            Accuracy estimate
        """
        return (inference_results["softmax_confidence"] > self.atc).float().mean().item()

    def get_atc_ts_estimate(self, inference_results: Dict) -> float:
        """
        Get ATC estimate with global TS
        Args:
            inference_results: dict with 'softmax_confidence_after_temperature' key
        Returns:
            Accuracy estimate
        """
        if isinstance(inference_results["softmax_confidence_after_temperature"], np.ndarray):
            return (inference_results["softmax_confidence_after_temperature"] > self.atc_ts).astype(float).mean()
        return (inference_results["softmax_confidence_after_temperature"] > self.atc_ts).float().mean().item()

    def get_atc_cs_ts_estimate(self, inference_results: Dict) -> float:
        """
        Get ATC estimate with classwise TS and class-wise ATC thresholds.
        """
        return (
            (inference_results["softmax_confidence_cs_ts"] > self.atc_cs_ts[inference_results["predictions"]])
            .float()
            .mean()
            .item()
        )

    def get_doc_estimate(self, inference_results: Dict) -> float:
        """
        Get DoC estimate without temperature scaling.
        """
        return (inference_results["softmax_confidence"].mean() - self.doc).item()

    def get_doc_ts_estimate(self, inference_results) -> float:
        """
        Get DoC estimate with temperature scaling.
        """
        return (inference_results["softmax_confidence_after_temperature"].mean() - self.doc_ts).item()

    def get_doc_cs_ts_estimate(self, inference_results: Dict) -> float:
        """
        Get DoC estimate with classwise TS and class-wise ATC thresholds.
        """
        return (
            (inference_results["softmax_confidence_cs_ts"] - self.doc_cs_ts[inference_results["predictions"]])
            .mean()
            .item()
        )

    def get_atc_dist_estimate(self, inference_results: Dict, kept_by_distance: np.ndarray) -> float:
        """
        Get ATC-Dist (or ATC-DistCS) estimate without TS.

        Args:
            inference_results: Dict with 'softmax_confidence' key, as produced by run_inference function.
            kept_by_distance: np.ndarray[bool] of shape [n_samples,]. Boolean array indicating whether to
                                keep or reject each sample according to pre-compute distance checker.
        """
        return (kept_by_distance & (inference_results["softmax_confidence"].numpy() > self.atc)).mean()

    def get_atc_ts_dist_estimate(self, inference_results: Dict, kept_by_distance: np.ndarray) -> float:
        """
        Get ATC-Dist (or ATC-DistCS) estimate with global TS.

        Args:
            inference_results: Dict with 'softmax_confidence_after_temperature' key, as produced
                                by run_inference function.
            kept_by_distance: np.ndarray[bool] of shape [n_samples,]. Boolean array indicating whether to
                                keep or reject each sample according to pre-compute distance checker.
        """
        if isinstance(inference_results["softmax_confidence_after_temperature"], np.ndarray):
            return (
                ((inference_results["softmax_confidence_after_temperature"] > self.atc_ts) & kept_by_distance)
                .astype(float)
                .mean()
            )
        return (
            ((inference_results["softmax_confidence_after_temperature"] > self.atc_ts) & kept_by_distance)
            .float()
            .mean()
            .item()
        )

    def get_atc_cs_ts_dist_estimate(self, inference_results: Dict, kept_by_distance: np.ndarray) -> float:
        """
        Get ATC-Dist (or ATC-DistCS) estimate with class-wise TS.

        Args:
            inference_results: Dict with 'softmax_confidence_cs_ts' key, as produced by run_inference function.
            kept_by_distance: np.ndarray[bool] of shape [n_samples,]. Boolean array indicating whether to
                                keep or reject each sample according to pre-compute distance checker.
        """
        return (
            (
                (inference_results["softmax_confidence_cs_ts"] > self.atc_cs_ts[inference_results["predictions"]])
                & kept_by_distance
            )
            .float()
            .mean()
            .item()
        )
