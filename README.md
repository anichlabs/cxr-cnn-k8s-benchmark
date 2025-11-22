````markdown
# Comparing Custom and Transfer-Learning CNN Models for Chest X-ray Classification  
### Evaluating Performance and Scalability with Kubernetes Orchestration

This repository contains the full code and experimental artefacts for a chest X-ray (CXR) classification project that compares a custom CNN (Victorio) against transfer-learning baselines (MobileNetV2, EfficientNet-B0, ResNet-50).  

The core focus is data analytics:

* systematic comparison of architectures and initialisation domains (ImageNet versus CXR)
* rigorous evaluation on held-out test sets (CheXpert_small and TBX11K)
* explainability analysis with Grad-CAM and LIME, including an Agreement Index (IoU)
* basic operational metrics under deployment (CPU versus GPU latency in Kubernetes)

The deployment and MLOps components exist to support the analytics story: they provide a controlled environment to measure performance and scalability, rather than being an independent product.


## 1. Repository structure

High-level layout:

* `00_setup_project.ipynb`  
  Initial environment checks, path configuration and basic sanity tests.

* `notebooks/`  
  * `01_data_eda.ipynb`  
    Exploratory data analysis for CheXpert_small and TBX11K.
  * `02_training_baseline.ipynb`  
    Baseline models and first training experiments.
  * `03_model_training.ipynb`  
    Final training runs for all CNNs with consistent configuration.
  * `04_evaluation_results.ipynb`  
    Test-set evaluation, cross-model comparison and tables for the report.
  * `05_explainability.ipynb`  
    Grad-CAM, LIME and Agreement Index (IoU) between both methods.
  * `06_deployment_metrics.ipynb`  
    CPU versus GPU inference latency using the deployed FastAPI service on Kubernetes.

* `src/`  
  * `dataset.py`  
    Dataset classes and loaders for CheXpert_small and TBX11K.
  * `transforms.py`  
    Centralised image preprocessing (resize, cropping, normalisation) used across notebooks and deployment.
  * `loaders.py`  
    High-level dataloader helpers for training, validation and test splits.

* `experiments/`  
  * `checkpoints/`  
    Trained model weights  
    * `mobilenet_v2_*_best.pt`  
    * `efficientnet_b0_*_best.pt`  
    * `resnet50_*_best.pt`  
    * `victorio_*_best.pt`
  * `pools/`  
    * `preprocessing_config.json` (image size, normalisation, class map)  
    * `source_pool_rel.csv`, `test_split.csv`
  * `results/`  
    Per-model metrics exported as JSON.
  * `tables/`  
    * `agreement_index.csv` (Grad-CAM versus LIME IoU results)

* `deployment/`  
  * `api/`  
    * `app/main.py`  
      FastAPI inference service (CPU and GPU aware, model loading, `/predict`, `/health`, `/metadata`).  
    * `app/models/victorio.py`  
      Implementation of the custom Victorio CNN.  
    * `Containerfile.cpu` / `Containerfile.gpu`  
      Podman container definitions for CPU and GPU images.  
  * `k8s/inference/`  
    * `deployment.cpu.yaml`  
    * `deployment.gpu.yaml`  
    * `service.cpu.yaml`  
    * `service.gpu.yaml`  
    * `block-gpu-in-default.yaml` (resource quota to prevent other pods consuming the GPU)  
    * `config/preprocessing_config.json` (mounted into the inference containers)

* `reports/`  
  * `figures/`  
    Confusion matrices, accuracy plots, and other figures used in the written thesis.  
  * `tables/`  
    * `cross_model_summary.csv`  
    * `test_evaluation_summary.(csv|json)`  
    * `k8s_operational_metrics.(csv|json)`
  * `thesis/`  
    Source documents related to the research proposal and ethics approval.

* `environment.yaml`  
  Conda environment specification for the main Python environment.

* `README.md`  
  This file.


## 2. Data and problem setting

The project uses two public CXR datasets (not redistributed in this repository):

* CheXpert (downsampled subset, referred to as CheXpert_small)  
* TBX11K

The final classification task is a three-class problem:

* Normal  
* Pneumonia  
* Tuberculosis (TB)

`notebooks/01_data_eda.ipynb` documents:

* class distributions and label balance  
* image resolution and aspect ratios  
* domain differences between CheXpert_small and TBX11K  
* decisions taken for data splits and harmonisation

All subsequent notebooks and the deployment stack rely on the same data splits and preprocessing configuration defined under `experiments/pools/`.


## 3. Models and training pipeline

CNN architectures:

* MobileNetV2 (transfer learning from ImageNet and CXR initialisation)  
* EfficientNet-B0 (transfer learning from ImageNet and CXR initialisation)  
* ResNet-50 (transfer learning from ImageNet and CXR initialisation)  
* Victorio (custom CNN designed in this project)

Key analytics questions:

* How does a custom CNN compare against standard transfer-learning backbones for this CXR task?  
* Does pretraining on ImageNet or CXR-trained weights provide a measurable advantage?  
* Are performance differences stable across validation and test splits?

Training is carried out mainly in:

* `02_training_baseline.ipynb`  
  First pass training and early observations.  
* `03_model_training.ipynb`  
  Refined training procedure with harmonised configuration across models.

Training details (learning rate, optimiser, loss function, regularisation, early stopping) are documented within the notebooks and reflected in the exported metrics under `experiments/results/`.


## 4. Evaluation metrics and cross-model comparison

The main evaluation is captured in `04_evaluation_results.ipynb`.

Metrics include:

* per-class precision, recall and F1-score  
* macro-averaged metrics  
* overall accuracy  
* confusion matrices for each model and domain

Aggregated results are exported to:

* `reports/tables/test_evaluation_summary.(csv|json)`  
* `reports/tables/cross_model_summary.csv`

These tables provide the basis for the quantitative comparison in the thesis:

* which model achieves the best overall accuracy and F1  
* how much variance exists between architectures  
* how pretraining strategy (ImageNet versus CXR) affects performance  


## 5. Explainability and Agreement Index

Explainability is implemented in `05_explainability.ipynb` using:

* Grad-CAM  
* LIME

The analytics objective is twofold:

* provide qualitative and quantitative evidence that models attend to plausible CXR regions  
* measure the agreement between Grad-CAM and LIME for the same image and model

The Agreement Index is defined as:

* Intersection-over-Union (IoU) between Grad-CAM and LIME salient regions on a per-image basis  
* aggregated across images and models into summary statistics  
* exported to `experiments/tables/agreement_index.csv`

This creates an extra analytic layer beyond raw accuracy, linking model performance to spatial consistency of saliency maps.


## 6. Deployment and operational metrics (analytics view)

To study scalability and operational behaviour, the final model (Victorio) is deployed as a FastAPI service and run on both CPU and GPU via Kubernetes (Minikube).  

Notebook `06_deployment_metrics.ipynb` performs:

* health checks for CPU and GPU endpoints  
* repeated prediction calls with the same CXR image  
* measurement of per-request latency on both backends  
* aggregation into summary statistics (mean, standard deviation, number of samples)  
* export to:

  * `reports/tables/k8s_operational_metrics.csv`  
  * `reports/tables/k8s_operational_metrics.json`

This provides a minimal but concrete link between:

* model-level performance (accuracy, F1, IoU)  
* system-level performance (inference time on CPU versus GPU)  
* infrastructure choices (local GPU node, containerisation, orchestration)

In the written thesis, these metrics are intended for the Results and Discussion sections, not as a standalone operations study.


## 7. Reproducibility: environment and training

The core experimental pipeline can be reproduced on a machine with sufficient CPU, RAM, disk and (optionally) an NVIDIA GPU.

### 7.1 Create the environment

1. Create the conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate msc-cxr
````

2. Verify that key libraries (PyTorch, torchvision, etc.) import correctly by running:

   ```bash
   python -c "import torch, torchvision; print(torch.__version__)"
   ```

### 7.2 Prepare data

1. Obtain CheXpert and TBX11K according to their respective licenses.

2. Place them under:

   ```text
   data/chexpert/...
   data/tbx11k/...
   ```

3. Ensure that the directory layout matches the expectations in `01_data_eda.ipynb` and `src/dataset.py`.

### 7.3 Re-run training and evaluation

In order:

1. `00_setup_project.ipynb`
2. `01_data_eda.ipynb`
3. `02_training_baseline.ipynb`
4. `03_model_training.ipynb`
5. `04_evaluation_results.ipynb`
6. `05_explainability.ipynb`
7. `06_deployment_metrics.ipynb` (requires the deployment stack to be running, see next section)

Each notebook is written to be executed top-to-bottom. Parameters and paths are centralised in `experiments/pools/preprocessing_config.json` and `src/transforms.py` wherever possible.

## 8. Inference API and Kubernetes deployment

This section gives a concise operational overview. Full details are in `deployment/api/` and `deployment/k8s/inference/`.

### 8.1 Building the containers (Podman)

From the `deployment/api/` directory:

* CPU image:

  ```bash
  podman build -f Containerfile.cpu -t cxr-cpu .
  ```

* GPU image:

  ```bash
  podman build -f Containerfile.gpu -t cxr-gpu .
  ```

### 8.2 Loading images into Minikube

1. Save and load the images into the Minikube container runtime. For example:

   ```bash
   podman save -o cxr-cpu.tar localhost/cxr-cpu:latest
   podman save -o cxr-gpu.tar localhost/cxr-gpu:latest

   minikube image load cxr-cpu.tar
   minikube image load cxr-gpu.tar
   ```

2. Verify:

   ```bash
   minikube image ls | grep cxr
   ```

### 8.3 CPU and GPU deployments

Apply the namespace and manifests:

```bash
kubectl create namespace inference

kubectl apply -f deployment/k8s/inference/block-gpu-in-default.yaml
kubectl apply -f deployment/k8s/inference/deployment.cpu.yaml
kubectl apply -f deployment/k8s/inference/service.cpu.yaml
kubectl apply -f deployment/k8s/inference/deployment.gpu.yaml
kubectl apply -f deployment/k8s/inference/service.gpu.yaml
```

The GPU deployment assumes:

* an NVIDIA GPU is available to Minikube
* the NVIDIA device plugin is installed and advertising `nvidia.com/gpu`

### 8.4 Port-forward and test

In separate terminals:

* CPU:

  ```bash
  kubectl port-forward deployment/cxr-inference-cpu 8000:8000 -n inference
  ```

* GPU:

  ```bash
  kubectl port-forward deployment/cxr-inference-gpu 8001:8000 -n inference
  ```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

Prediction example:

```bash
IMG="path/to/example_cxr.png"

curl -X POST http://localhost:8000/predict \
  -H "accept: application/json" \
  -F "file=@${IMG}"
```

## 9. Intended usage

This repository is designed for:

* examiners and supervisors who wish to inspect code, metrics and artefacts that support the written MSc thesis
* researchers and students who want a self-contained example of:

  * a small but complete deep learning experiment on medical imaging
  * structured comparison of custom and transfer-learning CNNs
  * integration of explainability techniques (Grad-CAM, LIME)
  * basic Kubernetes-based deployment with reproducible operational measurements

It is not intended as a clinical tool and must not be used to make any diagnostic decisions.

## 10. Acknowledgements

* CheXpert and TBX11K dataset creators and maintainers.
* CCT College Dublin for providing the MSc Data Analytics programme context.
* Open-source libraries and tools used throughout (PyTorch, torchvision, FastAPI, Uvicorn, Podman, Minikube, Kubernetes and others).

```
```
