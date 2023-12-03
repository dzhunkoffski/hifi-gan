import torch
from torch import nn
import torch.nn.functional as F

from src.loss.adversarial_loss import GeneratorLoss, DiscriminatorLoss
from src.loss.mel_loss import MelLoss
from src.loss.feature_loss import FeatureLoss

class GLoss(nn.Module):
    def __init__(self, mel_config: dict, lambda_mel: float, lambda_fm: float) -> None:
        super().__init__()

        self.adv_loss = GeneratorLoss()
        self.lambda_mel = lambda_mel
        self.mel_loss = MelLoss(**mel_config)
        self.lambda_fm = lambda_fm
        self.fm_loss = FeatureLoss()
    
    def forward(self, 
                wav_generated, wav_real, 
                mpd_features_generated, mpd_features_real, 
                msd_features_generated, msd_features_real,
                mpd_d_out_generated, mpd_d_out_real,
                msd_d_out_generated, msd_d_out_real, **batch
                ):
        adv_loss = self.adv_loss(mpd_d_out_generated, mpd_d_out_real) + self.adv_loss(msd_d_out_generated, msd_d_out_real)
        feature_map_loss = self.fm_loss(mpd_features_generated, mpd_features_real) + self.fm_loss(msd_features_generated, msd_features_real)
        mel_loss = self.mel_loss(wav_generated, wav_real)

        return adv_loss + self.lambda_fm * feature_map_loss + self.lambda_mel * mel_loss

class DLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adv_loss = DiscriminatorLoss()
    
    def forward(self, 
                mpd_d_out_generated,
                msd_d_out_generated, **batch):
        adv_loss = self.adv_loss(mpd_d_out_generated) + self.adv_loss(msd_d_out_generated)
        return adv_loss
