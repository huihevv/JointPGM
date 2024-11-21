# JointPGM

## Introduction

JointPGM focuses on the joint probabilistic modeling of intra/inter-series transitional shift for robust multivariate time series forecasting. 

## Preparation

1. Install Pytorch (>=1.11.0) and other necessary dependencies.
```
pip install -r requirements.txt
```
2. All the six benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1IJj9SYLyUc1qhjY6ns2VxmoivpcB3twe/view?usp=sharing). Exchange_rate is placed in the folder `./dataset` as an example. 

## Training scripts

We provide the JointPGM experiment scripts and hyperparameters of all benchmark datasets under the folder `./scripts`.

```bash
bash ./scripts/electricity/JointPGM.sh
bash ./scripts/ETTh1/JointPGM.sh
bash ./scripts/ETTm2/JointPGM.sh
bash ./scripts/exchange_rate/JointPGM.sh
bash ./scripts/ili/JointPGM.sh
bash ./scripts/metr/JointPGM.sh
```
