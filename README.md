# hifi-gan
DL-AUDIO homework. HiFiGan implementation

## Datasets
Audio:
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null --show-progress
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
