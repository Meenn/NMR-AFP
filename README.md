# NMR-AFP
This repository is the official PyTorch implementation of
"Integrating Non-local Memory-Augmented Reconstruction and Appearance-Motion Frame Prediction for Video Anomaly Detection"

## Environment
```
python==3.6
pytorch==1.5.1
mmcv-full==1.3.1
mmdet==2.11.0
scikit-learn==0.23.2
edflow==0.4.0
PyYAML==5.4.1
tensorboardX==2.4
```

## Data preparation
Please follow the [read.me](./pre_process/readme.md) to prepare the dataset.

## Training and Evaluation

### Training
First, to train the NL_ML_MemAE , run:
```python
$ python nl_ml_memAE_train.py
```
Then, to train the AMTAE with reconstructed flows, run:
```python
$ python trian.py
```
Finally, finetune the whole framework as:
```python
$ python finetune.py
```

### Evaluation
To evaluation the trained model, run:
```python
$ python eval.py --model_save_path=[model_save_path] 
                 --cfg_file=[cfg_file] 
```
You can download the pretrained weights of NMR-AFP for Ped2, Avenue and ShanghaiTech datasets 
from [here](https://drive.google.com/drive/folders/1W4MdP6Dnxy6O0x1ZcpPWM3n2T2OYobfh?usp=sharing).

## Results

|     Method     | UCSD Ped2 | CUHK Avenue | ShanghaiTech |
| :------------: | :-------: | :---------: | :----------: |
|    NMR-AFP     |   99.6%   |    91.5%    |    77.0%     |

## Acknowledgements

This code is heavily borrowed from [hf2vad](https://github.com/LiUzHiAn/hf2vad). Thanks LiUzHiAn for their contributions.
