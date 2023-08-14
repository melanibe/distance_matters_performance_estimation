from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from default_paths import DATA_MNIST
from torchvision.transforms import ToTensor, Compose, Resize
from data_handling.augmentations import ExpandChannels
from torchvision.datasets import MNIST, SVHN


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=128, num_workers=12, shuffle=True, *args, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.preprocess = Compose([ToTensor(), ExpandChannels()])

    def setup(self, stage=None):
        self.dataset_train = MNIST(root=DATA_MNIST, train=True, transform=self.preprocess, download=True)
        self.dataset_val = MNIST(root=DATA_MNIST, train=False, transform=self.preprocess, download=True)

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

    def test_dataloader(self):
        raise NotImplementedError

    def get_all_ood_dataloaders(self):
        svhn = SVHN(root=DATA_MNIST, split="test", transform=Compose([ToTensor(), Resize(28)]), download=True)
        return [
            (
                "svhn",
                DataLoader(svhn, self.batch_size, shuffle=False, num_workers=self.num_workers),
            ),
        ]

    @property
    def num_classes(self):
        return 10
