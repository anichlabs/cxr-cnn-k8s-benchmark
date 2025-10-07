"""
loader.py
---------
Utilities for building stratified PyTorch DataLoaders from the unifid CXR dataset.

Validated originally in '02_training_baseline.ipynb' and refactored here to
enable consistent, reproducible dataset splits across all experiments.

Responsibilities:
- Split the full dataset into stratified train/validation subsets (e.g.,80/20).
- Attach the correct transforms to each subset (train vs eval).
- Return DataLoader objects optimised for GPU throughput.
"""

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from src.dataset import CXRThreeClassDataset

def build_dataloaders(
    source_pool_rel,
    class_to_index: dict,
    chexpert_root,
    tbx_root,
    img_size: int,
    train_transforms,
    eval_transforms,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Construct stratified train/validation DataLoaders using shared transforms.

    Args:
        source_pool_rel (pd.DataFrame): Unified pool (relative paths + labels + source tags).
        class_to_index (dict): Mapping from class names to numeric indices.
        chexpert_root (Path): Root directory for CheXpert dataset.
        tbx_root (Path): Root directory for TBX11K dataset.
        img_size (int): Target image size for transforms.
        train_transforms, eval_transforms: Preprocessing pipelines.
        seed (int): Random seed for reproducible splitting.
        batch_size (int): Batch size for both loaders.
        num_workers (int): Number of parallel data-loading workers.
        pin_memory (bool): Enable CUDA page-locked memory.

    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # 1. Prepare stratified indices.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    y = source_pool_rel["Class"]

    for train_idx, val_idx in splitter.split(source_pool_rel, y):
        train_items = source_pool_rel.iloc[train_idx]
        val_items = source_pool_rel.iloc[val_idx]

    # 2. Instantiate datasets.
    train_dataset = CXRThreeClassDataset(
        items=train_items,
        class_to_index=class_to_index,
        transforms=train_transforms,
        chexpert_root=chexpert_root,
        tbx_root=tbx_root,
    )

    val_dataset = CXRThreeClassDataset(
        items=val_items,
        class_to_index=class_to_index,
        transforms=eval_transforms,
        chexpert_root=chexpert_root,
        tbx_root=tbx_root,
    )

    # 3. Build DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader