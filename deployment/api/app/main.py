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

from fastapi import FastAPI, UploadFile
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import json

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
    description="Serves chest X-ray classification models trained in the Jupyter "
                "notebooks of the project."
)


############################################################
# 2. Define where configuration will live inside container #
############################################################
# When the Podman container is built, mount:
#   experiments/pools/preprocessing_config.json
# into:
#   /app/config/preprocessing_config.json
#
# The API loads this ONCE at startup.
CONFIG_PATH = Path("/app/config/preprocessing_config.json")


############################################################
# 3. Load configuration when the API starts                #
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

    global cfg, img_size, mean, std
    # Parse the config JSON file.
    cfg = json.loads(CONFIG_PATH.read_text())

    # Required configuration values.
    img_size = int(cfg["img_size"]) # For now, ImageNet mode
    mean = cfg["imagenet_mean"]     # Allow swithing to CXR later.
    std  = cfg["imagenet_std"]

    # No model is loaded yet: that comes in a later step.


###########################################################
# 4. Health endpoint                                      #
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
    return{"status": "ok"}


###########################################################
# 5. Metadata endpoint                                    #
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
# 6. Prediction endpoint                                  #
###########################################################
@app.post("/predict")
async def predict(file: UploadFile):
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
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Apply transforms and add batch dimension
    x = tfm(img).unsqueeze(0)

    # ----------------------------------------------------
    # 3. Dummy prediction
    # ----------------------------------------------------
    # Later, replace this with:
    #    model = load_state_dict(...)
    #    pred  = model(x).argmax(dim=1)
    dummy_prediction = torch.tensor([0])

    # ----------------------------------------------------
    # 4. Return results in JSON
    # ----------------------------------------------------
    return {
        "prediction_index": int(dummy_prediction.item()),
        "class_map": cfg["class_to_index"]
    }
