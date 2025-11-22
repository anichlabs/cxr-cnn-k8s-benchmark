
# Research: Comparing Custom and Transfer-Learning CNN Models for Chest X-ray Classification: Evaluating Performance and Scalability with Kubernetes Orchestration.

This repository is organised to ensure modularity, reproducibility, and clarity for academic research.
---
```text

## Directory Layout

msc-sept-24-ft-cohort-capstone-chrisanich/
├── 00_setup_project.ipynb
├── data/
│   ├── chexpert/
│   └── tbx11k/
├── deployment/
│   ├── api/
│   │   ├── app/
│   │   ├── Containerfile.cpu
│   │   ├── Containerfile.gpu
│   │   ├── requirements-cpu.txt
│   │   ├── requirements-gpu.txt
│   │   └── wheels/
│   ├── k8s/
│   │   └── inference/
│   │       ├── deployment.cpu.yaml
│   │       ├── deployment.gpu.yaml
│   │       ├── service.cpu.yaml
│   │       ├── service.gpu.yaml
│   │       ├── block-gpu-in-default.yaml
│   │       └── config/preprocessing_config.json
│   │        
├── environment.yaml
├── experiments/
│   ├── checkpoints/
│   ├── pools/
│   ├── results/
│   └── tables/
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_training_baseline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_results.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_deployment_metrics.ipynb
├── README.md
├── reports/
├── src/
└── terminal_commands.md

```

### Exact environment (works best on NVIDIA GPUs with CUDA 12.8+)
`conda env create -f environment.yml`

### Or, for CPU-only / other GPUs (auto-selects correct PyTorch wheel)
- `conda create -n thesis python=3.10 pip`
- `conda activate thesis`
- `pip install torch torchvision torchaudio grad-cam --index-url https://download.pytorch.org/whl/cpu`
- `pip install matplotlib seaborn pandas scikit-learn opencv-python pillow tqdm pyyaml`
