import os
from typing import Callable, Optional, Tuple, Union

import fast_transformers.builders as ftb
import fast_transformers.feature_maps as ftfm
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as tg
import wandb
from miniproject.configuration import Config
from miniproject.metrics import AverageLoss
from omegaconf import OmegaConf
from pytorch_lightning.metrics import Accuracy, ConfusionMatrix

from .graph_attention import GraphAttentionModel
from .performer import PerformerModel


class ShapeNetModel(pl.LightningModule):
    hparams: Config

    def __init__(self, conf: Config):
        super().__init__()

        # Inner model
        self.model = build_model(conf)

        # Metrics
        num_classes = conf.architecture.out_features
        self.metrics_train = torch.nn.ModuleDict({"accuracy": Accuracy()})
        self.metrics_val = torch.nn.ModuleDict(
            {
                "loss": AverageLoss(),
                "accuracy": Accuracy(),
                "confusion": ConfusionMatrix(num_classes),
            }
        )
        self.metrics_test = torch.nn.ModuleDict(
            {
                "loss": AverageLoss(),
                "accuracy": Accuracy(),
                "confusion": ConfusionMatrix(num_classes),
            }
        )

        # Bookeeping
        self.save_hyperparameters(conf)

    def forward(self, batch: tg.data.Batch) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch: tg.data.Batch, batch_idx):
        logits = self(batch)

        categories = batch.category.to(self.device)
        loss = torch.nn.functional.cross_entropy(logits, categories, reduction="mean")

        # Note: this works in single-gpu and in DistributedDataParallel,
        # if using DataParallel use training_step_end
        accuracy = self.metrics_train.accuracy(logits, categories)
        self.log("loss/train", loss, on_step=True)
        self.log("acc/train", accuracy, on_step=True)

        # with open(f'{os.getpid()}.txt','a') as f:
        #     print(os.getpid(), loss.device, 'train_step', loss.item(),file=f)

        return {"loss": loss, "logits": logits, "targets": categories}

    def validation_step(self, batch, batch_idx):
        num_graphs = torch.tensor(batch.num_graphs, device=batch.x.device)
        logits = self(batch)
        categories = batch.category.to(self.device)
        loss = torch.nn.functional.cross_entropy(logits, categories)

        # Note: this works in single-gpu and in DistributedDataParallel,
        # if using DataParallel use validation_step_end
        self.metrics_val.accuracy.update(logits, categories)
        self.metrics_val.confusion.update(logits, categories)
        self.metrics_val.loss.update(loss, num_graphs)

    def validation_epoch_end(self, outputs):
        self.log("acc/val", self.metrics_val.accuracy.compute())
        self.log("loss/val", self.metrics_val.loss.compute())
        # self.logger.experiment.log(
        #     {
        #         "confusion/val": wandb.Table(
        #             columns=list(range(self.hparams.architecture.out_features)),
        #             rows=list(range(self.hparams.architecture.out_features)),
        #             data=self.val_confusion.compute().cpu().numpy(),
        #         )
        #     },
        #     commit=False,
        # )

    def test_step(self, batch: tg.data.Batch, batch_idx):
        num_graphs = torch.tensor(batch.num_graphs, device=batch.x.device)
        logits = self(batch)
        categories = batch.category.to(self.device)
        loss = torch.nn.functional.cross_entropy(logits, categories)

        # Note: this works in single-gpu and in DistributedDataParallel,
        # if using DataParallel use test_step_end
        self.metrics_test.accuracy.update(logits, categories)
        self.metrics_test.confusion.update(logits, categories)
        self.metrics_test.loss.update(loss, num_graphs)

    def test_epoch_end(self, outputs):
        self.log("acc/test", self.metrics_test.accuracy.compute())
        self.log("loss/test", self.metrics_test.loss.compute())
        # self.log("confusion/test", self.test_confusion.compute())
        # self.logger.experiment.log(
        #     {
        #         "confusion/test": wandb.Table(
        #             columns=list(range(self.hparams.architecture.out_features)),
        #             rows=list(range(self.hparams.architecture.out_features)),
        #             data=self.test_confusion.compute().cpu().numpy(),
        #         )
        #     },
        #     commit=False,
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.optimizer.learning_rate,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        return optimizer


def build_model(conf: Config):
    if conf.architecture.model == "gat":
        return GraphAttentionModel(conf.architecture)
    if conf.architecture.model == "performer":
        return PerformerModel(conf.architecture)
    raise ValueError(f"Invalid model: {conf.architecture.model}")
