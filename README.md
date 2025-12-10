# Disentangling Latent Distortions: A Transformer U-Net Diffusion Model for Robust Time Series Anomaly Detection

This repository provides the official implementation of TUCAD (Transformer U-Net Cross-Attentive Diffusion), a diffusion-based model designed to improve robustness against latent distortions and enhance anomaly detection performance in multivariate time series.

TUCAD integrates a Transformer U-Net architecture with cross-attentive fusion to capture long-range temporal dependencies while preserving fine-grained local structures—addressing a major limitation in conventional reconstruction-based anomaly detection models.

---

## Overview

TUCAD is designed to:

- Mitigate latent-space distortion and cross-variable interference
- Leverage diffusion modeling for stable reconstruction
- Utilize cross-attention to align encoder–decoder representations
- Achieve state-of-the-art anomaly detection performance across diverse datasets

## Description
### Code
project/
├── data/
├── src/
│ ├── condition_denoiser_models
│ ├── denoiser_models
 |  ├── loss_functions
 |  ├── transformers
 |  ├── dataset_utils
 |  └── utils_eval
├── train.py # main training script
├── test.py # main training script
├── run.sh # training/testing launcher
├── requirements.txt
└── README.md

### Dataset
The model is evaluated on six benchmark datasets:
- PSM
- SMD
- MSL
- SMAP
- NIPS-TS-SWAN
- SWaT

Due to the 30 MB submission size limit, we contain dataset download links.
---


## Get Started
1. Training 
python train.py --model_name {Denoiser_name} --dataset SMD --window_size 20 --stride 1 --batch_size 64 --epochs 10 --T 500

2. Test
python test.py --model_name {Denoiser_name} --dataset SMD --window_size 20 --overlap True --batch_size 64 --epochs 10 --T 500