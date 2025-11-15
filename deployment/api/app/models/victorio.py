"""
victorio.py
===========

Reconstruction of the custom CNN ("Victorio") defined in Notebook 03.
Checkpoints trained for this architecture require the model to be
rebuild exactly with the same layers before weights can be loaded.

Architecture:
- Conv -> ReLU -> MaxPool
- Conv -> ReLU -> MaxPool
- AdaptiveAvgPool to fixed 4x4
- Flatten + Linear for classification
"""

import torch.nn as nn

def build_victorio(num_classes: int):
    """
    Create the Victorio CNN architecture with the same structure
    used during training in 03_model_training.ipynb.

    Args:
        num_classes (int): number of output classes.
    
    Returns:
        nn.Module: instantiated CNN model.
    """
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        
        nn.AdaptiveAvgPool2d((4, 4)),  # Force output to 4x4, regardless of input.
                                       # AdaptativeAvgPool2d dynamically squeezes each feature map
                                       # to 4x4 spatial dimensions. This is what architectures like
                                       # ResNet and EfficientNet do. It ensures the fully connected
                                       # layer always receives a fixed-length vector.
        nn.Flatten(),
        
        nn.Linear(32 * 4 * 4, num_classes),
    )