import os
from typing import Any, Callable, Optional

import torch

from pytorch_lightning.metrics.metric import Metric


class AverageLoss(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor, count: torch.Tensor):
        # print(os.getpid(), value.device, 'update', value.item(), count.item())
        self.value += value * count
        self.count += count

    def compute(self):
        # print(os.getpid(), self.value.device, 'compute', self.value.item(), self.count.item())
        return self.value / self.count
