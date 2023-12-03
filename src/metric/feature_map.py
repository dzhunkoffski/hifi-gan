import torch
from torch import nn
import torch.nn.functional as F

from src.base.base_metric import BaseMetric
from src.loss.feature_loss import FeatureLoss

class FeatureMapMetric(BaseMetric):
    def __init__(self):
        super().__init__()

        self.feature_loss = FeatureLoss()

    def __call__(self, mpd_features_generated, mpd_features_real, 
                msd_features_generated, msd_features_real,**batch):
        return (self.feature_loss(mpd_features_generated, mpd_features_real) + self.feature_loss(msd_features_generated, msd_features_real)).item()