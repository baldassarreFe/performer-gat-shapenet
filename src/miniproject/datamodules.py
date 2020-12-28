from pathlib import Path
from typing import Union
import torch
import torch_geometric as tg
import pytorch_lightning as pl


class ShapeNetDataModule(pl.LightningDataModule):
    in_features = 3 + 3
    categories = tuple(tg.datasets.ShapeNet.category_ids.keys())

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_points: int,
        num_workers: int,
        batch_size: int,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.num_points = num_points
        self.batch_size = batch_size

    def prepare_data(self):
        tg.datasets.ShapeNet(root=self.data_dir, split="train")
        tg.datasets.ShapeNet(root=self.data_dir, split="val")
        tg.datasets.ShapeNet(root=self.data_dir, split="test")

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            self.ds_train = tg.datasets.ShapeNet(
                root=self.data_dir,
                split="train",
                transform=tg.transforms.Compose(
                    [
                        tg.transforms.FixedPoints(
                            self.num_points, replace=False, allow_duplicates=True
                        ),
                        tg.transforms.RandomTranslate(0.01),
                        tg.transforms.RandomScale((0.9, 1.1)),
                        tg.transforms.RandomRotate(10, axis=0),
                        tg.transforms.RandomRotate(180, axis=1),
                        tg.transforms.RandomRotate(10, axis=2),
                    ]
                ),
            )
            self.ds_val = tg.datasets.ShapeNet(
                root=self.data_dir,
                split="val",
                transform=tg.transforms.Compose(
                    [
                        tg.transforms.FixedPoints(
                            self.num_points, replace=False, allow_duplicates=True
                        ),
                    ]
                ),
            )

        if stage is None or stage == "test":
            self.ds_test = tg.datasets.ShapeNet(
                root=self.data_dir,
                split="test",
                transform=tg.transforms.Compose(
                    [
                        tg.transforms.FixedPoints(
                            self.num_points, replace=False, allow_duplicates=True
                        ),
                    ]
                ),
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=tg.data.dataloader.Collater(follow_batch=[]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=tg.data.dataloader.Collater(follow_batch=[]),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=tg.data.dataloader.Collater(follow_batch=[]),
        )
