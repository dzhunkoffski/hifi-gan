import logging
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PAD_MEL = -11.5129251

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    batch_wav = []
    batch_mel_spec = []

    max_spectrogram_len = 0

    for item in dataset_items:
        batch_wav.append(item['audio'].t())
        max_spectrogram_len = max(item['spectrogram'].size()[-1], max_spectrogram_len)
    
    # max_spectrogram_len = (max_spectrogram_len // 256) * 256 + 255
    for item in dataset_items:
        batch_mel_spec.append(
            F.pad(
                input=item['spectrogram'], pad=(0, max_spectrogram_len - item['spectrogram'].size()[-1]),
                mode='constant', value=PAD_MEL
            ).squeeze(0)
        )
    
    batch_wav = torch.permute(torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True, padding_value=0), (0, 2, 1))
    batch_mel_spec = torch.stack(batch_mel_spec)

    return {
        "wav_real": batch_wav,
        "spectrogram": batch_mel_spec
    }
    