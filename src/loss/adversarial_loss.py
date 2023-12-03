import torch
from torch import nn
import torch.nn.functional as F

class GeneratorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, generated, real):
        loss = 0
        for o_generated, o_real in zip(generated, real):
            loss += torch.mean((o_real - 1) ** 2) + torch.mean(o_generated ** 2)
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, generated):
        loss = 0
        for o_generated in generated:
            loss = loss + torch.mean((1 - o_generated) ** 2)
        return loss