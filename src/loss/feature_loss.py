import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from typing import List

class FeatureLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, fms_generated: List[List[Tensor]], fms_real: List[List[Tensor]]):
        loss = 0
        for d_g, d_r in zip(fms_generated, fms_real):
            for layer_g, layer_r in zip(d_g, d_r):
                loss = loss + F.l1_loss(layer_g, layer_r)

        # XXX: WHY (wtf with averaging)
        return 2 * loss