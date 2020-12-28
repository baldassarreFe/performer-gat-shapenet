import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from coolname import generate_slug
from omegaconf import OmegaConf


@dataclass
class Trainer:
    max_epochs: int = 3
    fast_dev_run: bool = False


@dataclass
class Architecture:
    model: str = "performer"
    num_layers: int = 1
    num_heads: int = 2
    in_features: int = 3 + 3
    hidden_features: int = 8
    out_features: int = 16
    knearest: Optional[int] = 6


@dataclass
class Optimizer:
    learning_rate: float = 1e-3
    weight_decay: float = 0


@dataclass
class Logging:
    run_name: str = generate_slug(3)
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    save_dir: str = "./runs"
    offline: bool = False
    save_period: int = 1
    save_top_k: int = 1
    log_every_n_steps: int = 5
    progress_bar: bool = sys.__stdout__.isatty()


@dataclass
class Other:
    gpus: int = 2
    seed: int = np.random.randint(0, np.iinfo(np.uint32).max)


@dataclass
class Data:
    data_root: str = "./data"
    num_points: int = 1000
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class Config:
    trainer: Trainer = Trainer()
    architecture: Architecture = Architecture()
    optimizer: Optimizer = Optimizer()
    logging: Logging = Logging()
    data: Data = Data()
    other: Other = Other()


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("conf", help="Configuration")

    subparser.add_argument(
        "files_or_overrides",
        type=str,
        metavar="arg",
        nargs="*",
        help="Config YAML files e.g. `conf.yaml` or overrides e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=parse_cli)


def parse_cli(files_or_overrides: List[str], **ignored_kwargs):
    files = []
    overrides = []
    for x in files_or_overrides:
        if "=" in x:
            overrides.append(x)
        elif x.endswith((".yaml", ".yml")):
            files.append(x)
        else:
            raise ValueError(f"Unrecognized: {x}")
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        *map(OmegaConf.load, files),
        OmegaConf.from_dotlist(overrides),
    )
    print(OmegaConf.to_yaml(conf, resolve=True))


if __name__ == "__main__":
    print(parse_cli())
