import torch
from torch import nn

from torch.nn.utils import weight_norm

from src.base.base_model import BaseModel

from typing import List

class InnerResBlock(nn.Module):
    def __init__(self, n_channels: int, k_r: int, D_r: List[int]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    weight_norm(nn.Conv1d(
                        in_channels=n_channels, out_channels=n_channels, kernel_size=k_r, dilation=D_r[l], 
                        padding='same'
                    ))
                )
            for l in range(len(D_r))]
        )
    
    def forward(self, input):
        x = torch.clone(input)
        for l in range(len(self.blocks)):
            x = self.blocks[l](x)
        x = x + input
        return x

class ResBlock(nn.Module):
    def __init__(self, n_channels: int, k_r: int, D_r: List[List[int]]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [InnerResBlock(n_channels=n_channels, k_r=k_r, D_r=D_r[m]) for m in range(len(D_r))]
        )
    
    def forward(self, input):
        for m in range(len(self.blocks)):
            input = self.blocks[m](input)
        return input


class MRF(nn.Module):
    def __init__(self, n_channels: int, k_r: List[int], D_r: List[List[List[int]]]):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResBlock(n_channels=n_channels, k_r=k_r[n], D_r=D_r[n]) for n in range(len(D_r))]
        )

    def forward(self, input):
        outputs = []
        output = torch.clone(input)
        for n in range(len(self.res_blocks)):
            output = self.res_blocks[n](output)
            outputs.append(torch.clone(output))
        output = torch.stack(outputs, dim=0).sum(dim=0)
        return output

class Generator(BaseModel):
    def __init__(self, in_features: int, k_u: List[int], h_u: int, k_r: List[int], D_r: List[List[List[int]]]):
        super().__init__()
        self.k_u = k_u
        self.h_u = h_u
        self.k_r = k_r
        self.D_r = D_r
        
        self.preconv = weight_norm(nn.Conv1d(in_channels=in_features, out_channels=h_u, kernel_size=7, dilation=1, padding=3))
        self.conv_transpose = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(
                in_channels=int(h_u / (2 ** l)), out_channels=int(h_u / (2 ** (l + 1))), 
                kernel_size=k_u[l], stride=k_u[l]//2, padding=(k_u[l] - k_u[l] // 2) // 2
            )) for l in range(len(k_u))
        ])
        self.mrf = nn.ModuleList(
            [MRF(n_channels=int(h_u / (2 ** (l + 1))), k_r=k_r, D_r=D_r) for l in range(len(k_u))]
        )
        self.leaky_relu = nn.LeakyReLU()
        self.post_conv = weight_norm(nn.Conv1d(in_channels=int(h_u / (2 ** (len(k_u)))), out_channels=1, kernel_size=7, padding=3))
        self.tanh = nn.Tanh()
    
    def forward(self, spectrogram, **batch):
        '''
        mel_spec: input mel spectrogramme
        '''
        x = self.preconv(spectrogram)

        for l in range(len(self.conv_transpose)):
            x = self.conv_transpose[l](x)
            x = self.mrf[l](x)
        x = self.leaky_relu(x)
        x = self.post_conv(x)
        x = self.tanh(x)
        
        return x