import torch
from torch import nn
import torch.nn.functional as F

from src.base.base_model import BaseModel

from torch.nn.utils import weight_norm

import math

class SubMPD(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()

        self.p = period
        self.blocks = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels=1, out_channels=2 ** 6, kernel_size=(5,1), stride=(3,1), padding=2)),
            weight_norm(nn.Conv2d(in_channels=2 ** 6, out_channels=2 ** 7, kernel_size=(5,1), stride=(3,1), padding=2)),
            weight_norm(nn.Conv2d(in_channels=2 ** 7, out_channels=2 ** 8, kernel_size=(5,1), stride=(3,1), padding=2)),
            weight_norm(nn.Conv2d(in_channels=2 ** 8, out_channels=2 ** 9, kernel_size=(5,1), stride=(3,1), padding=2)),
            weight_norm(nn.Conv2d(in_channels=2 ** 9, out_channels=2 ** 10, kernel_size=(5,1), padding=2)),
        ])
        self.leaky_relu = nn.LeakyReLU()
        self.post_conv = weight_norm(nn.Conv2d(in_channels=2 ** 10, out_channels=1, kernel_size=(7, 1)))
        
    
    def forward(self, x):
        batch_size, n_channels, timesteps = x.shape
        height = math.ceil(timesteps / self.p)
        width = self.p
        x = F.pad(x, (0, height * self.p - timesteps), "reflect", 0)
        x = x.view(batch_size, n_channels, height, width)

        feature_maps = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            x = self.leaky_relu(x)
            feature_maps.append(x)
        x = self.post_conv(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_maps

class MPD(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        self.sub_discriminators = nn.ModuleList([
            SubMPD(period=p) for p in periods
        ])
        
    def forward(self, x):
        feature_maps = []
        ys = []
        for i in range(len(self.sub_discriminators)):
            out, fmap = self.sub_discriminators[i](x)
            feature_maps.append(fmap)
            ys.append(out)
        return ys, feature_maps

class SubMSD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=41, stride=2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=41, stride=2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2))
        ])
        self.leaky_relu = nn.LeakyReLU()
        self.post_conv = weight_norm(nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        feature_maps = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            x = self.leaky_relu(x)
            feature_maps.append(x)
        x = self.post_conv(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_maps
    
class MSD(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            SubMSD(), SubMSD(), SubMSD()
        ])
        self.avg_pooling1 = nn.AvgPool1d(4, 2, padding=2)
        self.avg_pooling2 = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        outputs = []
        feature_maps = []
        
        out, fmap = self.sub_discriminators[0](x)
        outputs.append(out)
        feature_maps.append(fmap)

        x = self.avg_pooling1(x)
        out, fmap = self.sub_discriminators[1](x)
        outputs.append(out)
        feature_maps.append(fmap)

        x = self.avg_pooling2(x)
        out, fmap = self.sub_discriminators[2](x)
        outputs.append(out)
        feature_maps.append(fmap)

        return outputs, feature_maps