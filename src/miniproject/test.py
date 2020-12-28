import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as tg
import yaml
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from miniproject.datamodules import ShapeNetDataModule
from miniproject.models import ShapeNetModel

from .configuration import Config


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("test", help="Test")
    subparser.add_argument("ckpt_file", type=Path, help="Checkpoint file")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=test)


def parse_omegaconf(model, overrides):
    # Select allowed keys from model config
    filtered = OmegaConf.create()
    test_config_keys = [
        "other.gpus",
        "other.seed",
        "data.data_root",
        "data.num_workers",
        "data.batch_size",
    ]
    for key in test_config_keys:
        value = OmegaConf.select(model.hparams, key, throw_on_missing=True)
        OmegaConf.update(filtered, key, value, merge=False)
    # Set struct so overrides can not add extra keys
    OmegaConf.set_struct(filtered, True)
    conf = OmegaConf.merge(filtered, OmegaConf.from_dotlist(overrides))
    return conf


def test(ckpt_file, overrides, **ignored_kwargs):
    model = ShapeNetModel.load_from_checkpoint(ckpt_file)
    conf = parse_omegaconf(model, overrides)

    pl.seed_everything(conf.other.seed)
    pl.utilities.rank_zero_info(OmegaConf.to_yaml(conf))

    datamodule = ShapeNetDataModule(
        data_dir=conf.data.data_root,
        num_points=conf.data.num_points,
        num_workers=conf.data.num_workers,
        batch_size=conf.data.batch_size,
    )

    trainer = pl.Trainer(
        gpus=conf.other.gpus,
        accelerator="ddp",
        logger=None,
        default_root_dir=None,
        checkpoint_callback=None,
    )
    trainer.test(model, datamodule=datamodule, verbose=False)

    pl.utilities.rank_zero_info(
        f"acc/test {model.metrics.test.accuracy.compute().cpu().item():.2%}"
    )
    pl.utilities.rank_zero_info(
        f"loss/test {model.metrics.test.loss.compute().cpu().item():.4f}"
    )
    pl.utilities.rank_zero_info(
        "confusion/test\n%s",
        pd.DataFrame(
            model.metrics.test.confusion.compute().int().cpu().numpy(),
            index=ShapeNetDataModule.categories,
            columns=[c[:3] for c in ShapeNetDataModule.categories],
        ),
    )
