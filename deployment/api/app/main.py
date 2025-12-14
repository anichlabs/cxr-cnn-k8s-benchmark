""" main.py - FastAPI Interface Service.
=======================================

This file defines a *minimal but complete* inference API for the
Chest X-Ray classifier trained across notebooks 01-05.

Why this file exists:
---------------------
Notebook environments are excellent for research, exploration and explainability.
However, deployed machine-learning systems must be:

- reproducible
- self-contained
- versioned
- independent of notebooks
- usable by external clients (e.g. hospital systems).

FastAPI provides a clean and modern interface for that purpose.

This file will later:
- load the preprocessing_config.json
- load the trained model checkpoint (*.pt)
- apply the same transforms as training
- return predictions to external systems
"""

from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import json
from deployment.api.app.models.victorio import build_victorio

# This pulls the shared preprocessing builder from src/transforms.py.
from src.transforms import build_shared_transforms

###########################################################
# 1. Create the FastAPI application instance              #
###########################################################
# 'app' is the callable object Uvicorn runs inside the container.
# Uvicorn is an ASGI (async server gateway interface) compatible
# web server. It's the binding element that handles the web
# connections from the browser or API client. It allows FastAPI
# to serve the actual request.
# Uvicorn listens on a socket, receives the connection, does a bit
# processing and hands the request over to FastAPI, according to
# the ASGI interface.
app = FastAPI(
    title="CXR Inference API",
    version="1.0.0",
    description="Serves chest X-ray classification models trained in the Jupyter notebooks of the project."
)


############################################################
# 2. Model checkpoint path and architecture inference      #
############################################################
# This path will be mounted by Podman:
#   -v $(pwd)/experiments/checkpoints/model.pt:app/model/model.pt:ro
#  Allow container to start even if /app/model is empty (e.g. during debugging)
model_files = list(Path("/app/model").glob("*.pt"))
MODEL_PATH = model_files[0] if model_files else None


def parse_model_filename(ckpt_path: Path):
    """
    Parse clean checkpoint names of the form:
        mobilenet_v2_imagenet_best.pt
        efficientnet_b0_cxr_best.pt
        resnet50_imagenet_best.pt
        victorio_cxr_best.pt
    """

    stem = ckpt_path.stem
    parts = stem.split("_")

    # Architecture
    if stem.startswith("mobilenet_v2"):
        architecture = "mobilenet_v2"

    elif stem.startswith("efficientnet_b0"):
        architecture = "efficientnet_b0"

    elif stem.startswith("resnet50"):
        architecture = "resnet50"

    elif stem.startswith("victorio"):
        architecture = "victorio"

    else:
        raise ValueError(f"Cannot parse architecture from filename: {stem}")

    # Domain is always before "best"
    domain = parts[-2]  # "imagenet" or "cxr"

    return architecture, domain


############################################################
# 3. Define where configuration will live inside container #
############################################################
# When the Podman container is built, mount:
#   experiments/pools/preprocessing_config.json
# into:
#   /app/config/preprocessing_config.json
#
# The API loads this ONCE at startup.
CONFIG_PATH = Path("/app/config/preprocessing_config.json")


############################################################
# 4. Load configuration when the API starts                #
############################################################
@app.on_event("startup")
def load_config():
    """
    This function runs automatically ONCE when the API starts.
    It loads preprocessing parameters so inference matches training.

    Why this matters:
    -----------------
    - Image size, mean, std, and class index mapping must be identical
      to the settings used in the training notebooks.
    - If they differ even slightly, inference becomes invalid.
    """

    global cfg, img_size
    # Parse the config JSON file.
    cfg = json.loads(CONFIG_PATH.read_text())

    # Required configuration values.
    img_size = int(cfg["img_size"]) # For now, ImageNet mode

    # No model is loaded yet: that comes in a later step.

    # -------------------------------
    # 1. Select device (CPU o GPU)
    # -------------------------------
    # The service should run on CPU by default, but seamlessly
    # switch to GPU if the container is built with CUDA support
    # and executed with -gpus all (Podman) or proper NVIDIA runtime.
    #
    # torch.cuda.is_available() returns True only when:
    #   - the GPU variant of the image is used
    #   - the NVIDIA libraries are present
    #   - the runtime exposes the GPU to the container
    #
    # This keeps inference code simple and portable.
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------
    # 4. Detect checkpoint if mounted into the container
    # ---------------------------------------------------
    global ckpt_arch, ckpt_domain

    if MODEL_PATH is not None and MODEL_PATH.exists():
        ckpt_arch, ckpt_domain = parse_model_filename(MODEL_PATH)
    else:
        ckpt_arch, ckpt_domain = None, None

    # ---------------------------------------------------
    # 5. Adjust normalisation based on domain
    # ---------------------------------------------------
    global mean, std

    # always define defaults first
    mean = cfg["imagenet_mean"]
    std  = cfg["imagenet_std"]

    # override only if CXR model
    if ckpt_domain == "cxr":
        mean = cfg["cxr_mean"]
        std  = cfg["cxr_std"]

    # -------------------------------------------------------
    # 6. Build model architecture if a checkpoint is present
    # -------------------------------------------------------
    global model

    if ckpt_arch is None:
        model = None # Stay in dummy mode
        return
    
    num_classes = len(cfg["class_to_index"])

    if ckpt_arch == "mobilenet_v2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)

    elif ckpt_arch == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif ckpt_arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif ckpt_arch == "victorio":
        model = build_victorio(num_classes)

    else:
        raise ValueError(f"Unknown architecture: {ckpt_arch}")

    # ------------------------------------------
    # 7. Load weights and move model to device
    # ------------------------------------------
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # important for dropout/batchnorm

###########################################################
# 5. Health endpoint                                      #
###########################################################
@app.get("/health")
def health():
    """
    Purpose:
    -------
    Kubernetes, Podman, or any monitoring tool must know
    that the service is alive. This endpoint returns instantly.

    Used for:
    - readiness probes
    - liveness probes
    - quick debugging
    """
    return {"status": "ok"}


###########################################################
# 6. Metadata endpoint                                    #
###########################################################
@app.get("/metadata")
def metadata():
    """
    Returns configuration values used by the inference pipeline.
    Helpful for debugging and validation.
    """
    return {
        "image_size": img_size,
        "normalisation_mean": mean,
        "normalisation_std": std,
        "classes": cfg["class_to_index"]
    }


###########################################################
# 7. Prediction endpoint                                  #
###########################################################
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    This endpoint receives a chest X-ray image (JPG/PNG),
    applies the SAME preprocessing as in training,
    and returns a prediction.

    For now:
    --------
    We use a *dummy prediction* because model loading is a
    separate step (it needs a stable Containerfile).

    In the next section of Notebook 06:
    - load the real PyTorch model
    - push the model to GPU (if available)
    - return real predictions
    """

    # ----------------------------------------------------
    # 1. Load the image from the upload stream
    # ----------------------------------------------------
    img = Image.open(file.file).convert("RGB")

    # ----------------------------------------------------
    # 2. Recreate the preprocessing pipeline
    # ----------------------------------------------------
    # This MUST match:
    # - notebooks 02, 03, 04
    # - src/transforms.py logic
    # Build evaluation transforms from the shared transform builder.
    # Use cfg["transform_mode"] so the API matches the persisted config.
    _, tfm = build_shared_transforms(
        img_size=img_size,
        mean=mean,
        std=std,
        transform_mode=cfg.get("transform_mode", "center-crop")
    )

    # Apply transforms and add batch dimension
    x = tfm(img).unsqueeze(0)

    # ----------------------------------------------------
    # 3. Real prediction if model is available
    # ----------------------------------------------------
    if model is not None:
        model.eval()
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())

        idx_to_class = {v: k for k, v in cfg["class_to_index"].items()}
        pred_class = idx_to_class[pred_idx]

        return {
            "prediction_index": pred_idx,
            "prediction_class": pred_class,
            "architecture": ckpt_arch,
            "domain": ckpt_domain,
            "device": str(device),
            "class_map": cfg["class_to_index"]
        }

    # ----------------------------------------------------
    # 4. Fallback: no checkpoint mounted (dummy mode)
    # ----------------------------------------------------
    dummy_prediction = torch.tensor([0])
    return {
        "prediction_index": int(dummy_prediction.item()),
        "prediction_class": "dummy",
        "architecture": "none",
        "domain": "none",
        "device": str(device),
        "class_map": cfg["class_to_index"]
    }
