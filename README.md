# TUCAD
## 1. Title : Disentangling Latent Distortions: A Transformer U-Net Diffusion Model for Robust Time Series Anomaly Detection

This repository provides the official implementation of TUCAD (Transformer U-Net Cross-Attentive Diffusion), a diffusion-based model designed to improve robustness against latent distortions and enhance anomaly detection performance in multivariate time series.

TUCAD integrates a Transformer U-Net architecture with cross-attentive fusion to capture long-range temporal dependencies while preserving fine-grained local structures—addressing a major limitation in conventional reconstruction-based anomaly detection models.

----

## 2. Description
TUCAD (Transformer U-Net Cross-Attentive Diffusion) is a diffusion-based anomaly 
detection framework designed to improve robustness against latent-space distortion
and cross-variable interference in multivariate time series.

The model integrates a Transformer U-Net architecture with cross-attentive fusion
to capture long-range temporal dependencies while preserving fine-grained local
structures. By leveraging diffusion-based denoising, TUCAD achieves stable
reconstruction and robust anomaly scoring across diverse benchmark datasets.

This repository provides the complete implementation of TUCAD, including training
and inference scripts, model definitions, and evaluation utilities. In addition,
the exact preprocessed versions of all datasets used in the experiments are made
publicly available to ensure full reproducibility.

----

## 3. Code Information
The repository is organized to clearly separate the core components of the TUCAD
framework from optional or auxiliary implementations.

project\
├── data	#data directory\
├── src	#Model and utility implementations \
│   ├── condition_denoiser_models      # Main Diffusion denoising networks \
│   ├── denoiser_models                # Auxiliary Diffusion denoising networks (Optional) \
│   ├── loss_functions		# training loss definitions\
│   ├── transformers		# Transformer and attention modules\
│   ├── dataset_utils                  # Sliding window utilities\
│   └── utils_eval                     # Evaluation metrics\
├── train.py                            # Training script\
├── test.py                             # Inference script\
├── condition_diffusion.py              # Main Diffusion process \
├── diffusion.py                        # Auxiliary Diffusion process (Optional) \
├── run.sh\
├── requirements.txt\
└── README.md



## 4. Datasets Information
The proposed model is evaluated on six widely used benchmark datasets for
multivariate time series anomaly detection: PSM, SMD, MSL, SMAP, SWaT, and NIPS-TS-SWAN.

The original versions of these datasets are publicly available from their respective
repositories:
- PSM: https://github.com/eBay/RANSynCoders.git
- SMD: https://github.com/NetManAIOps/OmniAnomaly.git
- MSL / SMAP: https://github.com/khundman/telemanom.git
- SWaT: https://itrust.sutd.edu.sg/itrust-labs_datasets/

Most of the original repositories provide raw or partially processed data.
To ensure full reproducibility, we release the exact preprocessed versions used in our
experiments via a public Zenodo repository:

**Zenodo (preprocessed datasets): https://doi.org/10.5281/zenodo.18080582**

Regarding the NIPS-TS-SWAN dataset, an official centralized download link has not been
publicly released. Following common practice in prior anomaly detection benchmarks,
we include its preprocessed version in the same Zenodo repository to facilitate
consistent and reproducible evaluation.

----

## 5. Usage Instructions
### 1) Dataset Preparation
Download the preprocessed datasets from the Zenodo repository:
https://doi.org/10.5281/zenodo.18080582

Place the datasets under:
data/final_dataset/

** Please ensure that the dataset paths are correctly specified before execution. **

### 2) Environment Setup
Install dependencies with: \
pip install -r requirements.txt 

### 3) Training 
python train.py --model_name {Denoiser_name} --dataset {dataset} ...
```bash
python train.py --model_name {Denoiser_name} --dataset {dataset} --window_size 20 --stride 1 --batch_size 64 --epochs 10 --T 500
```


### 4) Test 
python test.py --model_name {Denoiser_name} --dataset {dataset} ...
```bash
python test.py --model_name {Denoiser_name} --dataset {dataset} --window_size 20 --overlap True --batch_size 64 --epochs 10 --T 500
```


----

## 6. Requirements

### 1) Libraries
torch>=1.12 \
torchvision\
numpy~=1.23.2\
pandas~=1.5.1\
scikit-learn~=1.1.2\
tqdm~=4.64.1\
tensorboardx~=2.5.1

### 2) Hardware recommendations:
- NVIDIA GPU (12GB+ recommended)
- CUDA 11+ (if using GPU acceleration)

----

## 7. Computing Infrastructure
- Operating System : window 10 
- CPU : Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
- GPU : NVIDIA RTX 3090 TURBO D6X 24GB
- Memory : 64GB RAM

----

## 8. Methodology Summary
TUCAD follows a diffusion-based reconstruction pipeline:
- Sliding-window preprocessing is applied to construct fixed-length input sequences.
- A forward diffusion process gradually adds Gaussian noise over T steps.
- The Transformer U-Net denoiser with cross-attention predicts either noise.
- The reverse diffusion process reconstructs clean representations.
- Reconstruction errors are computed for each timestamp.
- data-driven thresholding is used to determine anomalous timestamps.

## 9. Citation
If you use this code or the provided datasets in your research, please cite:

Jeon, S., et al.,
"Disentangling Latent Distortions: A Transformer U-Net Diffusion Model
for Robust Time Series Anomaly Detection", under review.


## 10. License and Contribution
This project is released under the MIT License.

Contributions are welcome. Please feel free to submit issues or pull requests
to improve the codebase or documentation.
----


