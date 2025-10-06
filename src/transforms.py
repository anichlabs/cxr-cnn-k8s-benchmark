"""
transforms.py
-------------
Shared preprocessing pipelines for BOTH datasets (CheXpert + TBX11K).

These pipelines were validated in '02_training_baseline.ipynb' and refactored here
to ensure consistent input preprocessing across all experiments.

Responsibilities:
- Enforce RGB input (3 channels).
- Resize and crop to the target image size defined in the preprocessing config.
- Apply ImageNet mean/std normalisation for compatibility with pretrained models.
- Include safe augmentation (horizontal flip) for frontal X-Rays.
"""

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

def build_shared_transforms(img_size: int, imagenet_mean: list, imagenet_std: list):
    """
    Build train and evaluation transforms based on shared parameters.

    Args:
        img_size (int): Target image size (from preprocessing_config.json)
        imagenet_mean (list[float]): ImageNet RGB channels means.
        imagenet_std (list[float]): Imagenet RGB channels std deviations.

    Returns:
        tuple: (train_transforms, eval_transforms)
    """
    # --- Training pipeline ---.
    train_transforms_shared = T.Compose([
        T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),  # Scale shortest side to img_size (preserve aspect).
                                                                                       # Smooth resize.
        T.CenterCrop(img_size), # Deterministic crop; keeps central anatomy.
        T.RandomHorizontalFlip(p=0.5), # Safe augmentation for frontal CXRs.
        T.ToTensor(), # HWC (Height, Width, Channel -> CHW (Channel, Height, Width), convert to tensor [0..1].
        T.Normalize(imagenet_mean, imagenet_std) # Standardise to ImageNet stats.
    ])
    # --- Evaluation pipeline ---.
    eval_transforms_shared = T.Compose([
        T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),  # Preserve aspect ratio.
        T.CenterCrop(img_size), # Deterministic crop for evaluation.
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transforms_shared, eval_transforms_shared