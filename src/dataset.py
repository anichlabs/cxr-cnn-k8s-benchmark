"""
dataset.py
----------
Unified three-class Dataset for Chest X-Ray classification.

This class was validated in '02_training_baseline.ipynb'
and refactored into a reusable module for later experiments.

Responsibilities:
- Build absolute paths dynamically for CheXpert and TBX11K.
- Convert grayscale CheXpert images to RGB to match ImageNet backbones.
- Apply preprocessing transforms (resize, crop, normalisation, augmentation).
- Return PyTorch tensors (image, label) ready for DataLoader consumption.
"""

# Remove 'import json' when moving the next code to script. It is unused inside the class. JSON reading happens in the notebook when the config is loaded).
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# --- Unified Dataset class (same logic) ---
class CXRThreeClassDataset(Dataset):
    """
    Provide a unified three-class CXR dataset with the fixed label order:
      0 : 'Pneumonia' (CheXpert frontal-only)
      1 : 'TB'        (TBX11K)
      2 : 'Normal'    (TBX11K)

    Absolute paths are built at access time depending on the "Source" column.
    This class implements the standard PyTorch Dataset API (len, getitem).
    """

    def __init__(self, items: pd.DataFrame, class_to_index: dict, transforms, chexpert_root: Path, tbx_root: Path):
        # Store the DataFrame of items (relative paths + labels + sources).
        # Resetting the index ensures sequential indices from 0...n-1 for safe access.
        self.items = items.reset_index(drop=True)

        # Store mapping from class names to integer indices (e.g. {TB: 1, ...}).
        self.class_to_index = class_to_index
        
        # Save preprocessing transforms (data augmentation / normalisation).
        self.transforms = transforms

        # Save root paths to both datasets: need to build absolute paths later.
        self.chexpert_root = chexpert_root
        self.tbx_root = tbx_root

    def __len__(self) -> int:
        # Required by PyTorch: return the total number of samples in the dataset.
        return len(self.items)

    def __getitem__(self, idx: int):
        # Fetch the metadata for a single sample (row from the DataFrame).
        row = self.items.iloc[idx]

        # Build the absolute file path:
        # - If from CheXpert, join with CheXpert root.
        # - If from TBX11K, join with tbx_root + "imgs" folder.
        abs_path = (
            self.chexpert_root / row["rel_path"]
            if row["Source"] == "chexpert"
            else self.tbx_root / "imgs" / row["rel_path"]
        )

        # Load the image and convert to RGB explicitly.
        # This ensures grayscale CheXpert images become 3-channel, consistent with            # TBX11K and ImageNet.
        img = Image.open(abs_path).convert("RGB")

        # Apply preprocessing transforms if provided (resize, normalise, augment).
        if self.transforms is not None:
            img = self.transforms(img)

        # Convert the class string (e.g., "TB") into its numeric index (0/1/2).
        # Map class string to numeric index. 
        label = self.class_to_index[row["Class"]]

        # Return a tuple (image tensor, label tensor) ready for PyTorch DataLoader.
        return img, torch.tensor(label, dtype=torch.long)