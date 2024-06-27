# JointPGM

This is the official codebase for the paper: [Robust Multivariate Time Series Forecasting with Joint Probabilistic Modeling of Intra/Inter-Series Transitional Shift](https://arxiv.org/pdf/xxxx.pdf). 

## Introduction

JointPGM focuses on the joint probabilistic modeling of intra/inter-series transitional shift for robust multivariate time series forecasting. 

## Preparation

1. Install Pytorch (>=1.11.0) and other necessary dependencies.
```
pip install -r requirements.txt
```
2. All the six benchmark datasets are placed in the folder `./dataset`.

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

## Citation

If you find this repo useful, please cite our paper. 

```
@article{xxx,
  title={Robust Multivariate Time Series Forecasting with Joint Probabilistic Modeling of Intra/Inter-Series Transitional Shift},
  author={xxx},
  journal={arXiv preprint arXiv:xxx},
  year={2024}
}
```