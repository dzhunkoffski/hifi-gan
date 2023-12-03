import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

class MelLoss(nn.Module):
    def __init__(self, 
                 sample_rate: int, n_fft: int, win_length: int, 
                 hop_length: int, f_min: float, f_max: float, n_mels: int, power: float) -> None:
        super().__init__()
        self.real_transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=power
        )
        self.generated_transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=power
        )
        self.loss = nn.L1Loss()

    def forward(self, wav_generated: torch.Tensor, wav_real: torch.Tensor):
        mel_real = torch.log(self.real_transform(wav_real) + 1e-5)
        mel_generated = torch.log(self.generated_transform(wav_generated) + 1e-5)
        return self.loss(mel_generated, mel_real)
