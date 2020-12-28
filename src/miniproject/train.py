import argparse
from pathlib import Path
from typing import List

import numpy as np
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
    subparser = parser.add_parser("train", help="Train")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=train)


def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


def train(conf_file, overrides, **ignored_kwargs):
    conf = parse_omegaconf(conf_file, overrides)
    pl.seed_everything(conf.other.seed)
    pl.utilities.rank_zero_info(OmegaConf.to_yaml(conf))

    model = ShapeNetModel(conf)
    datamodule = ShapeNetDataModule(
        data_dir=conf.data.data_root,
        num_points=conf.data.num_points,
        num_workers=conf.data.num_workers,
        batch_size=conf.data.batch_size,
    )
    logger = WandbLogger(
        name=conf.logging.run_name,
        id=conf.logging.run_name,
        save_dir=conf.logging.save_dir,
        project="miniproject",
        offline=conf.logging.offline,
        tags=conf.logging.tags,
        group=conf.logging.group,
        notes=conf.logging.notes,
    )
    # logger=CSVLogger(
    #     save_dir=conflogging..save_dir,
    #     name=conf.logging.run_name,
    # )

    trainer = pl.Trainer(
        max_epochs=conf.trainer.max_epochs,
        gpus=conf.other.gpus,
        accelerator="ddp",
        default_root_dir=Path(conf.logging.save_dir) / conf.logging.run_name,
        logger=logger,
        log_every_n_steps=conf.logging.log_every_n_steps,
        fast_dev_run=5 if conf.trainer.fast_dev_run else False,
        progress_bar_refresh_rate=1 if conf.logging.progress_bar else 0,
        callbacks=[
            EarlyStopping(monitor="acc/val", patience=5, mode="max"),
            ModelCheckpoint(
                monitor="acc/val",
                mode="max",
                save_last=True,
                save_top_k=conf.logging.save_top_k,
                period=conf.logging.save_period,
            ),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
