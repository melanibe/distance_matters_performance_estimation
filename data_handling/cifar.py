from typing import Any
import pl_bolts.datamodules as dm
import numpy as np
import requests
from pathlib import Path
import io
from torch.utils.data import Dataset, DataLoader
from default_paths import DATA_CIFARC, ROOT
from torchvision.transforms import ToTensor

ALL_CIFAR10C_PERTURBATIONS = [
    "brightness",
    "elastic_transform",
    "gaussian_blur",
    "impulse_noise",
    "motion_blur",
    "shot_noise",
    "speckle_noise",
    "contrast",
    "fog",
    "gaussian_noise",
    "jpeg_compression",
    "pixelate",
    "snow",
    "zoom_blur",
    "defocus_blur",
    "frost",
    "glass_blur",
    "saturate",
    "spatter",
]


class CIFAR10DataModule(dm.CIFAR10DataModule):
    def __init__(self, batch_size=128, **kwargs: Any) -> None:
        super().__init__(data_dir=ROOT / "data", batch_size=batch_size, **kwargs)

    def get_all_ood_dataloaders(self):
        all_eval_dataloaders = []
        for perturbation in ALL_CIFAR10C_PERTURBATIONS:
            for severity in range(5):
                name = f"{perturbation}_s{severity}"
                ds = CIFAR10C(name=perturbation, transform=self.val_transforms, severity=severity)
                all_eval_dataloaders.extend(
                    [(name, DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False))]
                )
        return all_eval_dataloaders


class CIFAR10C(Dataset):
    def __init__(self, transform, name, severity):
        super().__init__()
        assert 0 <= severity <= 4
        relevant_idx = np.arange(severity * 10000, (severity + 1) * 10000)
        self.transform = ToTensor()
        self.name = name
        self.data = np.load(DATA_CIFARC / f"{name}.npy")[relevant_idx]
        self.labels = np.load(DATA_CIFARC / "labels.npy")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]


def download_cifar10h_labels(root: str = ".") -> np.ndarray:
    """
    Pulls cifar10h label data stream and returns it in numpy array.
    """
    try:
        cifar10h_labels = np.load(Path(root) / "cifar10h-counts.npy")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy"
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            cifar10h_labels = np.load(io.BytesIO(response.content))
        else:
            raise ValueError("Failed to download CIFAR10H labels!")
    return cifar10h_labels
