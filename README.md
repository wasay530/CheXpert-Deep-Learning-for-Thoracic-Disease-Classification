# CheXpert: Deep Learning for Thoracic Disease Classification
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-DenseNet121-blue)]()
[![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red)]()

## Abstract
This repository contains a PyTorch implementation for the classification of 14 common thoracic diseases using the **CheXpert** dataset (Stanford Hospital). The project addresses the challenge of multi-label classification in medical imaging, specifically focusing on handling **label uncertainty** (noisy labels) and optimizing **DenseNet121** for high-resolution chest radiographs.

## The Dataset & Challenge
CheXpert consists of **224,316 chest radiographs** from 65,240 patients. The core challenge of this dataset is the presence of "Uncertain" labels (labeled as `-1`), alongside Positive (`1`) and Negative (`0`) ground truths.

**The Labeling Problem:**
Unlike standard binary classification, the data contains noise in the form of uncertainty.
* `1`: Positive
* `0`: Negative
* `-1`: Uncertain (Noisy/Missing information)

<p align="center">
<img src="https://github.com/wasay530/chexpert_dataset_pytorch/blob/1b5c7e42319063f0d99d60b8ca523cf180bff619/Pict.png" title="Dataset Metadata Structure" width="800" alt="Dataset Structure">
</p>

## ⚙️ Methodology & Architecture

### 1. Data Preprocessing & Uncertainty Handling
To address the "Noisy Label" challenge, this implementation utilizes **Label Mapping** strategies to convert uncertainty into actionable training signals:
* **U-Ones / U-Zeros:** Mapping uncertain labels to positive or negative based on clinical prevalence.
* **Resolution:** Images are downsampled to standard resolutions for computational efficiency while retaining pathology features.

### 2. Model Architecture
* **Backbone:** DenseNet121 (Pre-trained on ImageNet).
* **Loss Function:** Binary Cross Entropy (BCE) with Logits for multi-label stability.
* **Optimizer:** Adam with Learning Rate Scheduling (ReduceLROnPlateau).

## Installation & Usage

### Prerequisites
* Python 3.9+
* PyTorch
* Pandas, NumPy, Scikit-Learn

### 1. Clone the Repository
```bash
git clone [https://github.com/wasay530/chexpert_dataset_pytorch.git](https://github.com/wasay530/chexpert_dataset_pytorch.git)
cd chexpert_dataset_pytorch
