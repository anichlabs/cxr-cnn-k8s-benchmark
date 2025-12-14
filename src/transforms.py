"""
transforms.py
-------------
Shared preprocessing pipelines for BOTH datasets (CheXpert + TBX11K).

These pipelines were validated in '02_training_baseline.ipynb' and refactored here
to ensure consistent input preprocessing across all experiments.

Responsibilities:
- Enforce RGB input (3 channels).
- Resize and crop to the target image size defined in the preprocessing config.
- Apply configurable mean/std normalisation (ImageNet or CXR) for flexibility.
- Include safe augmentation (horizontal flip) for frontal X-Rays.
"""

# PIL is used because torchvision transforms operate on PIL images.
from PIL import Image

# torchvision.transforms is the standard high-level API
# for composing preprocessing pipelines.
from torchvision import transforms as T

# InterpolationMode makes resize behavior explicit and reproducible.
from torchvision.transforms import InterpolationMode

# functional gives low-level image ops (resize, pad, crop)
# needed to implement letterbox correctly.
from torchvision.transforms import functional as F


class LetterboxResizePad:
    """
    Letterbox resize + padding to a fixed square.

    Letterbox means:
    - preserve aspect ratio.
    - do NOT crop anatomy.
    - pad the remaining space to reach a square image.

    This is NOT available as a single built-in torchvision transform,
    so it needs to be implemented explicitly.
    """

    def __init__(self, img_size: int, fill: int = 0):
        # img_size is the final width and height (e.g. 256 x 256)
        self.img_size = int(img_size)

        # fill is the padding colour
        # 0 = black, which is stable for chest X-rays
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Defensive programming:
        # ensure the input is a PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"LetterboxResizePad expects PIL.Image, got {type(img)}"
            )

        # Original image width (w) and height (h)
        w, h = img.size

        # Compute scaling factor so the LONGEST side becomes img_size
        # This guarantees the resized image fits inside the square.
        scale = self.img_size / float(max(w, h))

        # Compute new width and height while preserving aspect ratio
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Resize the image
        # functional.resize expects size as [height, width]
        img = F.resize(
            img,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        # Compute how much padding is needed to reach img_size x img_size
        pad_w = self.img_size - new_w
        pad_h = self.img_size - new_h

        # Split padding evenly on both sides.
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        # Apply padding
        # Padding order is [left, top, right, bottom]
        img = F.pad(
            img,
            padding=[left, top, right, bottom],
            fill=self.fill,
        )

        # Final safety check:
        # rounding can sometimes cause off-by-one errors
        # so the final size is enforced explicitly.
        if img.size != (self.img_size, self.img_size):
            img = F.center_crop(
                img,
                output_size=[self.img_size, self.img_size],
            )

        # Return the processed image
        return img


def build_shared_transforms(img_size: int, mean: list, std: list, transform_mode: str = "center-crop"):
    """
    Build train and evaluation transforms based on shared parameters.

    Args:
        img_size (int): Target image size (from preprocessing_config.json)
        mean (list[float]): Per-channel mean for normalisation (ImageNet or CXR).
        std (list[float]): Per-channel std deviation for normalisation (ImageNet or CXR).
        transform_mode (str): Spatial preprocessing mode ("center-crop" or "letterbox").

    Returns:
        tuple: (train_transforms, eval_transforms)
    """

    # Normalise transform_mode to avoid silent configuration bugs.
    transform_mode = transform_mode.strip().lower()

    # Spatial preprocessing choice:
    # -----------------------------
    if transform_mode == "letterbox":
        # Letterbox: preserve full field of view, pad to square:
        train_spatial = [
            LetterboxResizePad(img_size=img_size, fill=0),  # Aspect-preserving resize + padding.
        ]

        eval_spatial = [
            LetterboxResizePad(img_size=img_size, fill=0),  # Same spatial logic for evaluation.
        ]

    elif transform_mode == "center-crop":
        # Center-crop: resize then crop central square.
        train_spatial = [
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),  # Scale shortest side.
            T.CenterCrop(img_size),  # Deterministic crop; keeps central anatomy.
        ]

        eval_spatial = [
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),  # Preserve aspect ratio.
            T.CenterCrop(img_size),  # Deterministic crop for evaluation.
        ]

    else:
        raise ValueError(f"Unsupported transform_mode: {transform_mode}")

    # --- Training pipeline ---.
    train_transforms_shared = T.Compose(
        train_spatial + [
            T.RandomHorizontalFlip(p=0.5),  # Safe augmentation for frontal CXRs.
            T.ToTensor(),  # HWC -> CHW, convert to tensor [0..1].
            T.Normalize(mean, std)  # Standardise to chosen normalisation source.
        ]
    )

    # --- Evaluation pipeline ---.
    eval_transforms_shared = T.Compose(
        eval_spatial + [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return train_transforms_shared, eval_transforms_shared