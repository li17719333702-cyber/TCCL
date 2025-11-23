# TCCL: Template Convolution Contrastive Learning for Unsupervised Deep Clustering of Rolling Bearing Faults

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This repository contains the official PyTorch implementation of the paper "TCCL: Template Convolution Contrastive Learning for Unsupervised Deep Clustering of Rolling Bearing Faults".

## 📖 Abstract

Unsupervised fault diagnosis of rolling bearings remains a critical challenge due to the scarcity of labeled data. Existing methods often struggle with **shortcut learning**, leading to mode collapse where different fault types are indistinguishable in the feature space.

**Template Convolution Contrastive Learning (TCCL)** is a novel self-supervised framework that bridges the gap between unsupervised anomaly detection and fine-grained multi-class fault clustering. By employing a dynamic template matching mechanism, TCCL forces the model to learn discriminative morphological patterns of specific faults, creating a highly **cluster-friendly** representation space.

Comprehensive experiments on **CWRU**, **SEU**, and **MFPT** datasets demonstrate that TCCL achieves clustering accuracies exceeding **99%**, significantly outperforming 14 state-of-the-art baselines including SimCLR, MoCo, and TimesNet.

## ✨ Key Features

- **Dynamic Template Matching**: A novel mechanism to capture fine-grained fault-specific waveform structures.
- **Cluster-Friendly Representation**: Effectively separates different fault modes without supervision.
- **Comprehensive Benchmark**: Includes implementations of 14 baseline methods (Traditional, Deep Clustering, Contrastive Learning).
- **Reproducibility**: Full support for random seeding and deterministic training.

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/li17719333702-cyber/TCCL.git
   cd TCCL
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Key Requirements:**
   - Python >= 3.8
   - PyTorch >= 2.0
   - NumPy, Pandas, Scikit-learn
   - Matplotlib, Seaborn
   - UMAP-learn

## 📂 Dataset Preparation

The project supports three public rolling bearing datasets. Please download and organize them as follows:

- **CWRU Dataset**: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- **SEU Dataset**: Southeast University Gearbox & Bearing Data
- **MFPT Dataset**: [Society for Machinery Failure Prevention Technology](https://www.mfpt.org/fault-data-sets/)

Update the dataset paths in the arguments when running scripts, or modify the default paths in the code.

## 🚀 Usage

### 1. Quick Start (Single File Implementation)

For a quick test or easy integration, use the standalone script `single_tccl.py` which contains the full TCCL implementation in one file.

```bash
# Run with default settings (simulated data or default paths)
python single_tccl.py

# Run with specific dataset paths
python single_tccl.py --cwru_path /path/to/CWRU --epochs 100

# Full options
python single_tccl.py --help
```

### 2. Comprehensive Benchmark

To reproduce the paper's results and compare with other methods, use `benchmark.py`.

```bash
# Run benchmark on CWRU dataset
python benchmark.py --dataset CWRU --data_root /path/to/CWRU --epochs 50

# Run benchmark on SEU dataset
python benchmark.py --dataset SEU --data_root /path/to/SEU --epochs 50

# Run on all datasets automatically (save results only)
python benchmark.py --all --epochs 50
```

**Supported Methods in Benchmark:**

- **Traditional**: Raw+UMAP, Handcrafted Features, PCA+K-Means
- **Deep Clustering**: DEC, JULE, SCAN
- **Contrastive (Classic)**: SimCLR, MoCo, BYOL
- **Contrastive (SOTA)**: SimSiam, TS2Vec, VICReg, TimesNet
- **Ours**: TCCL

## 📊 Results

TCCL consistently achieves state-of-the-art performance across multiple metrics.

| Method | CWRU (Acc) | SEU (Acc) | MFPT (Acc) |
|--------|------------|-----------|------------|
| **TCCL (Ours)** | **100%** | **100%** | **99.94%** |
| MoCo | 99.50% | 99.80% | 80.90% |
| SimCLR | 28.40% | 89.80% | 81.30% |
| DEC | 96.20% | 68.10% | 59.40% |

*(See the full results table in the paper)*

## 📁 Project Structure

```text
TCCL/
├── datasets/               # Dataset loading scripts (CWRU, SEU, MFPT)
├── models/                 # Model definitions
│   ├── baselines/          # Implementation of baseline methods
│   └── tccl_model.py       # TCCL core model
├── utils/                  # Utility functions (metrics, visualization)
├── figures/                # Paper figures and LaTeX templates
├── single_tccl.py          # Standalone implementation
├── benchmark.py            # Main benchmarking script
├── visualize_dataset.py    # Data visualization tools
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{li2025tccl,
  title={TCCL: Template Convolution Contrastive Learning for Unsupervised Deep Clustering of Rolling Bearing Faults},
  author={Li, Huashun and Wu, Weimin and Zou, Chengjian},
  journal={Measurement},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
