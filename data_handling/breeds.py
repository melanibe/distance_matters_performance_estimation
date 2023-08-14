from robustness.tools.breeds_helpers import make_living17, make_nonliving26, make_entity13, make_entity30
from robustness import datasets
from pytorch_lightning import LightningDataModule
from torchvision.transforms import ToTensor, Compose, Resize
from default_paths import BREEDS_INFO_DIR, DATA_IMAGENET


class BreedsDataModuleBase(LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8, shuffle=True, *args, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.preprocess = Compose([Resize((224, 224)), ToTensor()])  # compose is required by breeds API
        self.create_datasets()

    def create_datasets(self):
        ret = self.dataset_creation_fn(BREEDS_INFO_DIR, split="rand")
        self.superclasses, subclass_split, _ = ret
        train_subclasses, test_subclasses = subclass_split
        dataset_source = datasets.CustomImageNet(
            DATA_IMAGENET, train_subclasses, transform_train=self.preprocess, transform_test=self.preprocess
        )
        loaders_source = dataset_source.make_loaders(
            self.num_workers, self.batch_size, data_aug=self.shuffle, shuffle_val=False
        )
        self.train_loader_source, self.val_loader_source = loaders_source
        dataset_target = datasets.CustomImageNet(
            DATA_IMAGENET, test_subclasses, transform_train=self.preprocess, transform_test=self.preprocess
        )
        loaders_target = dataset_target.make_loaders(self.num_workers, self.batch_size, shuffle_val=False)
        _, self.val_loader_target = loaders_target

    def train_dataloader(self):
        return self.train_loader_source

    def val_dataloader(self):
        return self.val_loader_source

    def get_all_ood_dataloaders(self):
        return [("ood_val", self.val_loader_target)]

    @property
    def num_classes(self):
        print(f"Num classes {len(self.superclasses)}")
        return len(self.superclasses)


class Living17DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_living17


class NonLiving26DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_nonliving26


class Entity13DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_entity13


class Entity30DataModule(BreedsDataModuleBase):
    @property
    def dataset_creation_fn(self):
        return make_entity30
