# Uncertainty-Based Spatial-Temporal Attention for Online Action Detection

## Introduction
Code of our ECCV2022 paper "Uncertainty-Based Spatial-Temporal Attention for Online Action Detection"

![Framework](https://github.com/guoh11/Uncertainty-OAD/blob/main/f2-framework.png)

## Datasets and Preparation
* THUMOS14 [[Link](http://crcv.ucf.edu/THUMOS14/download.html)]
* TVSeries [[Link](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html)]
* HDD [[Link](https://usa.honda-ri.com/hdd)]

Use this [Repo](https://github.com/yjxiong/anet2016-cuhk) to extract features.

## Environment
* Python==3.7.6
* PyTorch==1.7.1 
* CUDA==11.2

## Training
```
python train.py
```

## Inference
```
python inference.py
```

## References
We use the code from these repos for the baseline methods:
* Temporal Recurrent Networks for Online Action Detection [[GitHub](https://github.com/xumingze0308/TRN.pytorch)]
* OadTR: Online Action Detection with Transformers [[GitHub](https://github.com/wangxiang1230/OadTR)]
* Long Short-Term Transformer for Online Action Detection [[GitHub](https://github.com/amazon-research/long-short-term-transformer)]

## Citations
Please cite our paper if it helps your research:
```
add BibTex here
```
