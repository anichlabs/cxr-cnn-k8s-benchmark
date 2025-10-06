"""
src package
-----------
Reusable components of the project pipeline.

This package centralises the code validated in '/notebooks/02_training_baseline.ipynb':
- dataset.py    : unified three-class Dataset (CXRThreeClassDataset).
- transforms.py : shared ImageNet preprocessing pipelines.
- loaders.py    : stratified DataLoader constructor.
"""

from .dataset import CXRThreeClassDataset
from .transforms import build_shared_transforms
from .loaders import build_dataloaders

__all__ = [
    "CXRThreeClassDataset",
    "build_shared_transforms",
    "build_dataloaders",
]