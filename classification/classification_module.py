from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


class ClassificationModule(pl.LightningModule):
    """
    A generic PL module for classification
    """

    def __init__(
        self,
        num_classes: int,
        encoder_name: str,
        train_transform_module: torch.nn.Module,
        val_transform_module: torch.nn.Module,
        lr: float = 1e-4,
        patience_scheduler: int = 10,
        metric_to_monitor: str = "Val/AUROC",
        metric_to_monitor_mode: str = "max",
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience_scheduler = patience_scheduler
        self.metric_to_monitor = metric_to_monitor
        self.metric_to_monitor_mode = metric_to_monitor_mode
        self.model = self.get_model()
        self.train_transform_module = train_transform_module
        # this is saving it to the model for inference time
        # it also ensure that you validate with the right transforms.
        self.model.preprocess = val_transform_module
        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["target"]
        else:
            x, y = batch[0], batch[1]
        if self.trainer.training:
            x = self.train_transform_module(x)  # to perform GPU batched data augmentation
        else:
            x = self.model.preprocess(x)
        return x, y

    def common_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        probas = torch.softmax(output, 1)
        return loss, probas, target

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        loss, probas, targets = self.common_step(batch, batch_idx)
        self.log("Train/loss", loss, on_epoch=True, on_step=True)
        self.train_probas.append(probas.detach().cpu())
        self.train_targets.append(targets.detach().cpu())
        if batch_idx == 0:
            data = batch[0]
            data = data.cpu().numpy()
            f, ax = plt.subplots(2, 5, figsize=(15, 5))
            ax = ax.ravel()
            for i in range(min(10, data.shape[0])):
                img = np.transpose(data[i], [1, 2, 0])
                img = (img - img.min()) / (img.max() - img.min())
                ax[i].imshow(img)
                ax[i].axis("off")
            self.logger.experiment.add_figure("train/inputs", f, global_step=self.current_epoch)
        if torch.isnan(loss):
            raise ValueError("Found loss Nan")
        return loss

    def on_train_epoch_start(self) -> None:
        self.train_probas = []
        self.train_targets = []

    def on_train_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.train_targets), torch.cat(self.train_probas)
        preds = torch.argmax(probas, 1)
        try:
            if self.num_classes == 2:
                self.log("Train/AUROC", roc_auc_score(targets, probas[:, 1]))
            else:
                self.log("Train/AUROC", roc_auc_score(targets, probas, average="macro", multi_class="ovr"))
        except ValueError:
            pass
        self.log("Train/Accuracy", accuracy_score(targets, preds))

        self.train_probas = []
        self.train_targets = []

    def on_validation_epoch_start(self) -> None:
        self.validation_probas = []
        self.validation_targets = []

    def on_validation_epoch_end(self, unused=None) -> None:
        targets, probas = torch.cat(self.validation_targets).int(), torch.cat(self.validation_probas)
        preds = torch.argmax(probas, 1)
        try:
            if self.num_classes == 2:
                self.log("Val/AUROC", roc_auc_score(targets, probas[:, 1]))
            else:
                self.log("Val/AUROC", roc_auc_score(targets, probas, average="macro", multi_class="ovr"))
        # For iWilds you may not have all the classes in the dataset so you can not compute ROC
        except ValueError:
            pass
        self.log("Val/Accuracy", accuracy_score(targets, preds))
        self.validation_probas = []
        self.validation_targets = []

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        loss, probas, targets = self.common_step(batch, batch_idx)
        self.log("Val/loss", loss, on_epoch=True, on_step=False)
        self.validation_probas.append(probas.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())

        if batch_idx == 0:
            preds = torch.argmax(probas, 1)
            data = batch[0]
            wrong_x, wrong_y = (
                data[targets != preds].cpu().numpy(),
                targets[targets != preds].cpu().numpy(),
            )
            f, ax = plt.subplots(2, 5, figsize=(15, 5))
            ax = ax.ravel()
            for i in range(min(10, wrong_x.shape[0])):
                img = np.transpose(wrong_x[i], [1, 2, 0])
                img = (img - img.min()) / (img.max() - img.min())
                ax[i].imshow(img)
                ax[i].set_title(wrong_y[i])
                ax[i].axis("off")
            self.logger.experiment.add_figure("val/failed", f, global_step=self.current_epoch)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = [torch.optim.Adam(params_to_update, lr=self.lr, weight_decay=self.weight_decay)]
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer[0], patience=self.patience_scheduler, mode=self.metric_to_monitor_mode, min_lr=1e-5
            ),
            "monitor": self.metric_to_monitor,
        }
        return optimizer, scheduler

    def get_model(self) -> torch.nn.Module:
        if self.encoder_name.startswith("resnet"):
            return ResNetBase(num_classes=self.num_classes, encoder_name=self.encoder_name)
        elif self.encoder_name.startswith("efficientnet"):
            return EfficientNetBase(num_classes=self.num_classes, encoder_name=self.encoder_name)
        elif self.encoder_name.startswith("densenet"):
            return DenseNet121(num_classes=self.num_classes, encoder_name=self.encoder_name)
        else:
            raise NotImplementedError


class ResNetBase(torch.nn.Module):
    def __init__(self, num_classes: int, encoder_name: str) -> None:
        super().__init__()
        match encoder_name:
            case "resnet50_pretrained":
                self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            case "resnet18_pretrained":
                self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            case "resnet18":
                self.net = models.resnet18(weights=None)
            case "resnet50":
                self.net = models.resnet50(weights=None)
            case _:
                raise ValueError(f"Encoder name {encoder_name} not recognised.")
        self.num_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(self.num_features, num_classes)
        self.num_classes = num_classes

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def classify_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)


class EfficientNetBase(torch.nn.Module):
    def __init__(self, num_classes: int, encoder_name: str) -> None:
        super().__init__()
        match encoder_name:
            case "efficientnet_v2_s_pretrained":
                self.net = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
                self.net.classifier = models.efficientnet_v2_s(weights=None, num_classes=num_classes).classifier
            case "efficientnet_v2_s":
                self.net = models.efficientnet_v2_s(weights=None, num_classes=num_classes)
            case "efficientnet_v2_l_pretrained":
                self.net = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
                self.net.classifier = models.efficientnet_v2_l(weights=None, num_classes=num_classes).classifier
            case "efficientnet_v2_l":
                self.net = models.efficientnet_v2_l(weights=None, num_classes=num_classes)
            case _:
                raise ValueError(f"Encoder name {encoder_name} not recognised.")
        self.num_classes = num_classes

    def get_features(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def classify_features(self, x):
        return self.net.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)


class DenseNet121(torch.nn.Module):
    def __init__(self, num_classes: int, encoder_name: str) -> None:
        super().__init__()
        match encoder_name:
            case "densenet121_pretrained":
                self.net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            case "densenet121":
                self.net = models.densenet121(None)
            case _:
                raise ValueError(f"Encoder name {encoder_name} not recognised.")
        self.num_features = self.net.classifier.in_features
        self.net.classifier = torch.nn.Linear(self.num_features, num_classes)
        self.num_classes = num_classes

    def get_features(self, x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def classify_features(self, x):
        return self.net.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)
