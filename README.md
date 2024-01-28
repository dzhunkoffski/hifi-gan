# hifi-gan
DL-AUDIO homework. Implementation of the model called HiFi-GAN which uses a mel-spectrogram as input and upsamples it through transposed convolutions until the length of the output sequence matches the temporal resolution of raw waveforms. More details on model, train configuration and results examples located in wandb [report](https://wandb.ai/dzhunkoffski/hifigan/reports/Report-HIFI-GAN--Vmlldzo2NjU2Nzk4).

## Train the model
```bash
python train.py --config src/configs/hifi-gan.json
```

## Run in Kaggle
You can also use `hifigan_train.ipynb` notebook if you want to train model in kaggle.
