# TUCAD
## Disentangling Latent Distortions: A Transformer U-Net Diffusion Model for Robust Time Series Anomaly Detection

This repository provides the official implementation of TUCAD (Transformer U-Net Cross-Attentive Diffusion), a diffusion-based model designed to improve robustness against latent distortions and enhance anomaly detection performance in multivariate time series.

TUCAD integrates a Transformer U-Net architecture with cross-attentive fusion to capture long-range temporal dependencies while preserving fine-grained local structures—addressing a major limitation in conventional reconstruction-based anomaly detection models.

---

## 1. Overview

TUCAD is designed to:

- Mitigate latent-space distortion and cross-variable interference
- Leverage diffusion modeling for stable reconstruction
- Utilize cross-attention to align encoder–decoder representations
- Achieve state-of-the-art anomaly detection performance across diverse datasets

## 2. Description
### 1) Code
project\
├── src\
│   ├── condition_denoiser_models\ # Diffusion denoising networks (main)
│   ├── denoiser_models\ # Diffusion denoising networks (sub)
│   ├── loss_functions\
│   ├── transformers\
│   ├── dataset_utils\ # Sliding window
│   └── utils_eval\ # Evaluation metrics 
├── train.py\ # Training script
├── test.py\ # Inference 
├── condition_diffusion.py\ # diffusion process (main)
├── diffusion.py\ # diffusion process (sub)
├── run.sh\ 
├── requirements.txt\
└── README.md\


### 2) Dataset
The model is evaluated on six benchmark datasets:
- PSM : https://github.com/eBay/RANSynCoders.git
- SMD : https://github.com/NetManAIOps/OmniAnomaly.git
- MSL : https://github.com/khundman/telemanom.git
- SMAP : https://github.com/khundman/telemanom.git
- NIPS-TS-SWAN
- SWaT : https://itrust.sutd.edu.sg/itrust-labs_datasets/

\
- Most of the above repositories provide raw or partially processed versions of the datasets.
For convenience and reproducibility, we provide the exact preprocessed versions used in our experiments 
via the dataset_links.txt file included with this submission.

- Regarding NIPS-TS-SWAN, the dataset is commonly referenced in time-series anomaly detection benchmarks, 
but an official centralized download link has not been published. 
Instead, the dataset is typically accessed through benchmark repositories or scripts provided by related research projects. 
Therefore, we include the processed version directly in our dataset links for reproducibility.


## 3. Requirements
Install dependencies with: \
pip install -r requirements.txt 

### 1) Hardware recommendations:
- NVIDIA GPU (12GB+ recommended)
- CUDA 11+ (if using GPU acceleration)

## 4. Computing Infrastructure
- operating system : window10 
- CPU : Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
- GPU : NVIDIA RTX 3090 TURBO D6X 24GB
- Memory : 64GB RAM


## 5. Get Started
### 1) Training 
python train.py --model_name {Denoiser_name} --dataset SMD --window_size 20 --stride 1 --batch_size 64 --epochs 10 --T 500

### 2) Test 
python test.py --model_name {Denoiser_name} --dataset SMD --window_size 20 --overlap True --batch_size 64 --epochs 10 --T 500


