```
text
➜  cxr-cnn-k8s-benchmark git:(main) ✗ tree -L 3
.
├── 00_setup_project.ipynb
├── data
│   ├── chexpert
│   │   ├── train
│   │   ├── train.csv
│   │   ├── valid
│   │   └── valid.csv
│   └── tbx11k
│       ├── annotations
│       ├── imgs
│       ├── lists
│       ├── README.md
│       ├── TBX11K_CVPR2020.pdf
│       └── teaser.jpg
├── deployment
│   ├── api
│   │   ├── app
│   │   ├── Containerfile.cpu
│   │   ├── Containerfile.gpu
│   │   ├── requirements-cpu.txt
│   │   ├── requirements-gpu.txt
│   │   └── wheels
│   └── k8s
│       └── inference
├── environment.yaml
├── experiments
│   ├── checkpoints
│   │   ├── efficientnet_b0_cxr_best.pt
│   │   ├── efficientnet_b0_cxr_pilot.pt
│   │   ├── efficientnet_b0_imagenet_best.pt
│   │   ├── efficientnet_b0_imagenet_pilot.pt
│   │   ├── mobilenet_v2_cxr_best.pt
│   │   ├── mobilenet_v2_cxr_pilot.pt
│   │   ├── mobilenet_v2_imagenet_best.pt
│   │   ├── mobilenet_v2_imagenet_pilot.pt
│   │   ├── resnet50_cxr_best.pt
│   │   ├── resnet50_cxr_pilot.pt
│   │   ├── resnet50_imagenet_best.pt
│   │   ├── resnet50_imagenet_pilot.pt
│   │   ├── victorio_cxr_best.pt
│   │   ├── victorio_cxr_pilot.pt
│   │   ├── victorio_imagenet_best.pt
│   │   └── victorio_imagenet_pilot.pt
│   ├── pools
│   │   ├── preprocessing_config.json
│   │   ├── source_pool_rel.csv
│   │   ├── test_split.csv
│   │   ├── train_split.csv
│   │   └── val_split.csv
│   ├── results
│   │   ├── cross_model_summary.json
│   │   ├── efficientnet_b0_cxr_best_metrics.json
│   │   ├── efficientnet_b0_cxr_pilot_metrics.json
│   │   ├── efficientnet_b0_cxr_quick_metrics.json
│   │   ├── efficientnet_b0_imagenet_best_metrics.json
│   │   ├── efficientnet_b0_imagenet_pilot_metrics.json
│   │   ├── efficientnet_b0_imagenet_quick_metrics.json
│   │   ├── mobilenet_v2_cxr_best_metrics.json
│   │   ├── mobilenet_v2_cxr_pilot_metrics.json
│   │   ├── mobilenet_v2_cxr_quick_metrics.json
│   │   ├── mobilenet_v2_imagenet_best_metrics.json
│   │   ├── mobilenet_v2_imagenet_pilot_metrics.json
│   │   ├── mobilenet_v2_imagenet_quick_metrics.json
│   │   ├── resnet50_cxr_best_metrics.json
│   │   ├── resnet50_cxr_pilot_metrics.json
│   │   ├── resnet50_cxr_quick_metrics.json
│   │   ├── resnet50_imagenet_best_metrics.json
│   │   ├── resnet50_imagenet_pilot_metrics.json
│   │   ├── resnet50_imagenet_quick_metrics.json
│   │   ├── victorio_cxr_best_metrics.json
│   │   ├── victorio_cxr_pilot_metrics.json
│   │   ├── victorio_cxr_quick_metrics.json
│   │   ├── victorio_imagenet_best_metrics.json
│   │   ├── victorio_imagenet_pilot_metrics.json
│   │   └── victorio_imagenet_quick_metrics.json
│   └── tables
│       └── agreement_index.csv
├── notebooks
│   ├── 01_data_eda.ipynb
│   ├── 02_training_baseline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_results.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_deployment_metrics.ipynb
├── README.md
├── reports
│   ├── figures
│   │   ├── Agreement Index Between Grad-CAM and LIME Regions.png
│   │   ├── Confusion matrices for ResNet-50 (ImageNet) and Victorio (CXR-custom) on the test set.png
│   │   ├── confusion_matrix_grid_2x4.png
│   │   ├── cross_model_comparison_small_multiples.png
│   │   ├── cross_model_comparison_train_val.png
│   │   ├── efficientnet_b0_cxr_best_confusion_matrix.png
│   │   ├── efficientnet_b0_imagenet_best_confusion_matrix.png
│   │   ├── explainability
│   │   ├── mobilenet_v2_cxr_best_confusion_matrix.png
│   │   ├── mobilenet_v2_imagenet_best_confusion_matrix.png
│   │   ├── resnet50_cxr_best_confusion_matrix.png
│   │   ├── resnet50_imagenet_best_confusion_matrix.png
│   │   ├── Sample_Images_from_Each_Class.png
│   │   ├── Sanity_Check_Verify_Shared_Transforms_Are_Applied.png
│   │   ├── test_accuracy_all_models.png
│   │   ├── Unified_3-Class_Dataset_Distribution.png
│   │   ├── victorio_cxr_best_confusion_matrix.png
│   │   ├── victorio_imagenet_best_confusion_matrix.png
│   │   └── web_images
│   ├── notebooks_pdf
│   │   ├── 01_data_eda.pdf
│   │   ├── 02_training_baseline.pdf
│   │   ├── 03_model_training-2.pdf
│   │   ├── 04_evaluation_results-1.pdf
│   │   ├── 05_explainability.pdf
│   │   └── 06_deployment_metrics.pdf
│   ├── tables
│   │   ├── cross_model_summary.csv
│   │   ├── k8s_operational_metrics.csv
│   │   ├── k8s_operational_metrics.json
│   │   ├── test_evaluation_summary.csv
│   │   └── test_evaluation_summary.json
│   └── thesis
│       ├── CCT Assessment Cover Page Christopher Anich 2023202 Capstone.docx
│       ├── CCT Assessment Cover Page Christopher Anich 2023202 Capstone.odt
│       ├── CCT Ethics Approval Application v3.doc
│       ├── CCT Ethics Approval Application v3.pdf
│       ├── CCTP517 IP created by students.docx
│       ├── CCTP517 IP created by students.pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration2.odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration30.pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration3.odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration4.odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration5.odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestrationdocx.docx
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration(final2323).pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration(finalllll).pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration(final).odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration(final).pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration.odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration (studying for thesis).docx
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration (studying for thesis).odt
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration (studying for thesis).pdf
│       ├── Comparing Custom and Transfer-Learning CNN Models for Chest X-Ray Classification: Performance and Scalability with Kubernetes Orchestration (thesis).docx
│       ├── Exemplar dissertation example.pdf
│       ├── LibreOffice Writer.odt
│       ├── Research Proposal.docx
│       ├── Research Proposal.pdf
│       ├── Some guidelines for writing your capstone thesis.pptx
│       ├── Text File.txt
│       ├── Thesis Title Page Template(2).docx
│       ├── Viva.odp
│       ├── Viva.pdf
│       └── Viva.pptx
├── src
│   ├── dataset.py
│   ├── __init__.py
│   ├── loaders.py
│   ├── __pycache__
│   │   ├── dataset.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── loaders.cpython-310.pyc
│   │   └── transforms.cpython-310.pyc
│   └── transforms.py
└── terminal_commands.md

30 directories, 133 files
➜  cxr-cnn-k8s-benchmark git:(main) ✗ ls
00_setup_project.ipynb  deployment        experiments  README.md  src
data                    environment.yaml  notebooks    reports    terminal_commands.md
➜  cxr-cnn-k8s-benchmark git:(main) ✗ cd deployment 
➜  deployment git:(main) ✗ tree -L 3
.
├── api
│   ├── app
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models
│   ├── Containerfile.cpu
│   ├── Containerfile.gpu
│   ├── requirements-cpu.txt
│   ├── requirements-gpu.txt
│   └── wheels
│       ├── filelock-3.20.0-py3-none-any.whl
│       ├── fsspec-2025.10.0-py3-none-any.whl
│       ├── jinja2-3.1.6-py3-none-any.whl
│       ├── markupsafe-3.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
│       ├── mpmath-1.3.0-py3-none-any.whl
│       ├── networkx-3.4.2-py3-none-any.whl
│       ├── numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
│       ├── nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl
│       ├── nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
│       ├── nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl
│       ├── nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl
│       ├── nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl
│       ├── nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl
│       ├── nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
│       ├── nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
│       ├── pillow-12.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│       ├── sympy-1.14.0-py3-none-any.whl
│       ├── torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
│       ├── torchaudio-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
│       ├── torchvision-0.24.1-cp310-cp310-manylinux_2_28_x86_64.whl
│       ├── triton-3.5.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
│       └── typing_extensions-4.15.0-py3-none-any.whl
└── k8s
    └── inference
        ├── block-gpu-in-default.yaml
        ├── config
        ├── deployment.cpu.yaml
        ├── deployment.gpu.yaml
        ├── service.cpu.yaml
        └── service.gpu.yaml

8 directories, 40 files
➜  deployment git:(main) ✗ 
```
