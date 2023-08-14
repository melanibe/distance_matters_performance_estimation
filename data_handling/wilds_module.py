from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from pytorch_lightning import LightningDataModule
from default_paths import DATA_WILDS_ROOT
from torchvision.transforms import ToTensor, Compose, Resize
from data_handling.augmentations import Standardize


class WILDSBase(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8, shuffle=True, **kwargs) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) -> None:
        raise NotImplementedError("setup() should be implemented in child class")

    def train_dataloader(self):
        if self.shuffle:
            return get_train_loader(
                "standard", self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
        return get_eval_loader(
            "standard", self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return get_eval_loader("standard", self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return get_eval_loader("standard", self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_all_ood_dataloaders(self):
        return [
            (
                "ood_val",
                get_eval_loader(
                    "standard", self.ood_val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
                ),
            ),
            (
                "ood_test",
                get_eval_loader(
                    "standard", self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
                ),
            ),
        ]


class WILDSCameLyon17(WILDSBase):
    def setup(self, stage=None) -> None:
        self.dataset = get_dataset(dataset="camelyon17", download=True, root_dir=DATA_WILDS_ROOT)
        self.train_dataset = self.dataset.get_subset("train", transform=ToTensor())
        self.val_dataset = self.dataset.get_subset("id_val", transform=ToTensor())
        self.test_dataset = self.dataset.get_subset("test", transform=ToTensor())
        self.ood_val_dataset = self.dataset.get_subset("val", transform=ToTensor())

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def wilds_class_name(self):
        return "wilds_camelyon"


class WILDSiCam(WILDSBase):
    def __init__(self, batch_size=16, **kwargs) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.preprocess = Compose([ToTensor(), Resize((224, 224))])

    def setup(self, stage=None) -> None:
        self.dataset = get_dataset(dataset="iwildcam", download=True, root_dir=DATA_WILDS_ROOT)
        self.train_dataset = self.dataset.get_subset("train", transform=self.preprocess)
        self.val_dataset = self.dataset.get_subset("id_val", transform=self.preprocess)
        self.test_dataset = self.dataset.get_subset("test", transform=self.preprocess)
        self.ood_val_dataset = self.dataset.get_subset("val", transform=self.preprocess)
        print(self.dataset._n_classes)

    @property
    def num_classes(self) -> int:
        return 182

    @property
    def wilds_class_name(self):
        return "wilds_icam"


class WILDSrr1(WILDSBase):
    def __init__(self, batch_size=16, **kwargs) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.preprocess = Compose([ToTensor(), Resize((224, 224)), Standardize()])

    def setup(self, stage=None) -> None:
        self.dataset = get_dataset(dataset="rxrx1", download=True, root_dir=DATA_WILDS_ROOT)
        self.train_dataset = self.dataset.get_subset("train", transform=self.preprocess)
        # Use the "in-distribution" test as val split since no id val split is available.
        self.val_dataset = self.dataset.get_subset("id_test", transform=self.preprocess)
        self.ood_val_dataset = self.dataset.get_subset("val", transform=self.preprocess)
        self.test_dataset = self.dataset.get_subset("test", transform=self.preprocess)
        print(self.dataset._n_classes)

    @property
    def num_classes(self) -> int:
        return 1139

    @property
    def wilds_class_name(self):
        return "wilds_rr1"


class WILDSFMoW(WILDSBase):
    def __init__(self, batch_size=32, **kwargs) -> None:
        super().__init__(batch_size=batch_size, **kwargs)

    def setup(self, stage=None) -> None:
        self.dataset = get_dataset(dataset="fmow", download=False, root_dir=DATA_WILDS_ROOT)
        self.train_dataset = self.dataset.get_subset("train", transform=ToTensor())
        self.val_dataset = self.dataset.get_subset("id_val", transform=ToTensor())
        self.ood_val_dataset = self.dataset.get_subset("val", transform=ToTensor())
        self.test_dataset = self.dataset.get_subset("test", transform=ToTensor())
        print(self.dataset._n_classes)

    @property
    def num_classes(self) -> int:
        return 62

    @property
    def wilds_class_name(self):
        return "wilds_fmow"
