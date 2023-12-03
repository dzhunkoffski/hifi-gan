import torch
from torch import nn
import torch.nn.functional as F

from src.base.base_metric import BaseMetric
from src.loss.mel_loss import MelLoss

class MelMetric(BaseMetric):
    def __init__(self, mel_config: dict):
        super().__init__()
        self.mel_loss = MelLoss(**mel_config)

    def __call__(self, wav_generated, wav_real, **batch):
        self.mel_loss = self.mel_loss.to(wav_generated.device)
        return self.mel_loss(wav_generated, wav_real).item()