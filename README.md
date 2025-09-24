
# Research: Comparing Custom and Transfer-Learning CNN Models for Chest X-ray Classification: Evaluating Performance and Scalability with Kubernetes Orchestration.

This repository is organised to ensure modularity, reproducibility, and clarity for academic research.
---
```text

## Directory Layout

thesis-project/
│
├── data/                # Raw datasets (CheXpert_small, TBX11K) – untouched
│   ├── chexpert/        
│   └── tbx11k/
│
├── notebooks/           # Jupyter notebooks (EDA, sanity checks, plots for thesis)
│   ├── 01_data_eda.ipynb
│   ├── 02_training_baseline.ipynb
│   └── 03_explainability.ipynb
│
├── src/                 # Reusable, modular code
│   ├── datasets/        # PyTorch Dataset & transforms
│   ├── models/          # Custom CNN + baselines (MobileNetV2, EfficientNet, ResNet)
│   ├── training/        # Training loops, evaluation metrics
│   ├── explainability/  # Grad-CAM utilities
│   └── utils/           # Configs, logging, helpers
│
├── experiments/         # Configs + results for each experiment
│   ├── exp1_customcnn.yaml
│   ├── exp2_mobilenet.yaml
│   └── logs/
│
├── reports/             # Outputs for thesis writing
│   ├── figures/         # ROC curves, confusion matrices, Grad-CAM heatmaps
│   ├── tables/          # Metrics, statistical tests
│   └── thesis/          # Draft sections
│
├── environment.yml      # Conda environment specification
└── README.md            # Documentation
```

### Exact environment (works best on NVIDIA GPUs with CUDA 12.8+)
`conda env create -f environment.yml`

### Or, for CPU-only / other GPUs (auto-selects correct PyTorch wheel)
- `conda create -n thesis python=3.10 pip`
- `conda activate thesis`
- `pip install torch torchvision torchaudio grad-cam --index-url https://download.pytorch.org/whl/cpu`
- `pip install matplotlib seaborn pandas scikit-learn opencv-python pillow tqdm pyyaml`
