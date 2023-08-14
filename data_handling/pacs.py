from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np

from default_paths import DATA_PACS


class PACSModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=12, shuffle=True, *args, **kwargs):
        super().__init__()
        self.preprocess = ToTensor()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:
        self.dataset_train = PACSDataset(
            split="train",
            domain="photo",
            transform=self.preprocess,
        )

        self.dataset_val = PACSDataset(
            split="test",
            domain="photo",
            transform=self.preprocess,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_all_ood_dataloaders(self):
        list_loaders = []
        for domain in ["cartoon", "art_painting", "sketch"]:
            list_loaders.append(
                (
                    domain,
                    DataLoader(
                        PACSDataset(split="test", domain=domain, transform=self.preprocess),
                        self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                    ),
                )
            )
        return list_loaders

    @property
    def num_classes(self):
        return 7


class PACSDataset(Dataset):
    def __init__(self, split, domain, transform=None) -> None:
        super().__init__()
        self.split = split
        self.domain = domain
        self.name = f"{domain}_{split}.hdf5"
        self.transform = transform

        with h5py.File(DATA_PACS / self.name, "r") as f:
            self.images = np.array(f["images"][:]).astype(np.uint8)
            self.labels = np.array(f["labels"][:]).astype(np.int64) - 1

    def __getitem__(self, idx):
        if self.transform is None:
            return self.images[idx], self.labels[idx]
        return self.transform(self.images[idx]), self.labels[idx]

    def __len__(self):
        return self.labels.size
