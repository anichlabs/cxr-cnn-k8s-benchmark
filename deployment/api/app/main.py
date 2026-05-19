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

import time
import os
import logging
from app.instrumentation import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    INFERENCE_DURATION_SECONDS,
    PREDICTIONS_TOTAL,
)
from fastapi import FastAPI, UploadFile, File, Query
from pathlib import Path
from PIL import Image
import torch
import json
from app.metrics import router as metrics_router
from app.models.victorio import build_victorio

# This pulls the shared preprocessing builder from src/transforms.py.
from src.transforms import build_shared_transforms

# To tell FastAPI to serve the demo/index.html file.
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.DEBUG)

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
# Record startup time so we can report uptime in /dashboard-metrics
APP_START_TIME = time.time()

app = FastAPI(
    title="CXR Inference API",
    version="1.0.0",
    description="Serves chest X-ray classification models trained in the Jupyter notebooks of the project."
)

##########################################################
# Prometheus HTTP metrics middleware                     #
##########################################################
# This middleware wraps every incoming HTTP request handled by FastAPI.
#
# Why a middleware?
# -----------------
# - It runs for ALL endpoints (/health, /predict, /metrics, etc.)
# - It allows to measure request-level behaviour consistently
# - It avoids duplicating metrics code inside each route
#
# What it measures here:
# ----------------------
# 1. Total number of HTTP requests, labelled by:
#    - HTTP method  (GET, POST, ...)
#    - URL path     (/health, /predict, ...)
#    - Status code  (200, 400, 500, ...)
#
# 2. End-to-end request latency:
#    - Time from request arrival to response generation
#    - Includes FastAPI routing, validation, inference, etc.
#
# These metrics are:
# - Collected automatically
# - Exposed via /metrics (Prometheus scrape endpoint)
# - Used later for dashboards and SLOs
#
# IMPORTANT:
# ---------
# This measures *API-level* latency.
# Model inference latency is measured separately inside /predict
# using INFERENCE_DURATION_SECONDS.
#############################################################
@app.middleware("http")
async def prom_middleware(request, call_next):
    # Start a high-resolution timer as soon as the request enters the app.
    start = time.perf_counter()

    # Pass the request down the middleware chain and into the route handler.
    response = await call_next(request)

    # Compute total request duration (seconds).
    duration = time.perf_counter() - start

    # Extract request metadata used as Prometheus labels.
    path = request.url.path # e.g. /health, /predict
    method = request.method # GET, POST, ...
    status = str(response.status_code)

    # Increment total HTTP request counter.
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()

    # Observe end-to-end request latency (histogram).
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
    
    logging.debug(
        f"Metrics updated: {method} {path} {status} in {duration:.4f}s"
    )
    # Return the original response unchanged.
    return response

app.include_router(metrics_router)

############################################################
# 2. Model checkpoint path and architecture inference      #
############################################################
# Model checkpoint filename parsing
#
# Checkpoints are expected to be mounted into /app/model at runtime
# (e.g. via Podman, Docker, or Kubernetes volume mounts).
#
# This function does NOT load the model.
# It only infers architecture and training domain from the filename
# to decide which model class and preprocessing to use later.

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

MODEL_PATH: Path | None = None
model = None
ckpt_arch = None
ckpt_domain = None
device = None
mean = None
std = None

############################################################
# 4. Lazy loading. Load on-demand per request              #
############################################################
# Load on-demand per request:
# - Lower memory cost (only one model in RAM at a time)
# - Slower first request (model loads from disk)
# - Can swap models without restarting
def load_model_from_config(model_name: str, device: torch.device):
    """
    Load a model checkpoint on-demand based on the model name in models_config.json

    Args:
        model_name: The key from models_config.json (e.g., "mobilenet_v2_cxr")
        device: torch.device (cpu or cuda)

    Returns:
        (model, metadata_dict) where metadata contains architecture, domain, checkpoint_path

    Raises:
         ValueError if model_name not found in config
    """

    # Load config file
    config_path = Path("/app/config/models_config.json")
    config = json.loads(config_path.read_text())

    # Look up model in available_models
    if model_name not in config["available_models"]:
        raise ValueError(f"Model '{model_name}' not found in config. Available: {list(config['available_models'].keys())}")

    model_info = config["available_models"][model_name]

    # Build full path to checkpoint
    checkpoint_filename = model_info["checkpoint"]
    checkpoint_base = config["checkpoint_base_path"]
    checkpoint_path = Path(checkpoint_base) / checkpoint_filename

    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Load preprocessing config to know num_classes
    preprocessing_config = json.loads(Path("/app/config/preprocessing_config.json").read_text())
    num_classes = len(preprocessing_config["class_to_index"])

    # Build model architectire based on config
    architecture = model_info["architecture"]

    # Instantiate the correct model architecture
    # Note: weights=None means you start with random weights
    # Load the trained weights from the checkpoint file below
    if architecture == "mobilenet_v2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        # Replace the final classification later to match our num_classes
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)

    elif architecture == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        # EfficientNet's classifier is a Sequential with th linear layer at index 1
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif architecture == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        # ResNet uses model.fc for the final fully connected layer
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif architecture == "victorio":
        from app.models.victorio import build_victorio
        model = build_victorio(num_classes)

    else:
        raise ValueError(f'Unknown architecture: {architecture}')

    # Load the trained weights from disk
    # map_location=device ensures we load to the right device (CPU or GPU)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the correct device and set to evaluation mode
    # eval() disables dropout and batch normalisation training behaviour
    model.to(device)
    model.eval()

    # Return the loaded model and metadata about what was loaded
    metadata = {
        "model_name": model_name,
        "architecture": architecture,
        "domain": model_info["domain"],
        "checkpoint_path": str(checkpoint_path),
        "num_classes": num_classes
    }

    return model, metadata

############################################################
# 5. Load configuration when the API starts                #
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

    global cfg, img_size, MODEL_PATH

    model_files = list(Path("/app/model").glob("*.pt"))
    MODEL_PATH = model_files[0] if model_files else None

    # Parse the config JSON file.
    cfg = json.loads(CONFIG_PATH.read_text())

    # Required configuration values.
    img_size = int(cfg["img_size"]) # Input size used during training.
    if img_size <= 0:
        raise ValueError("Invalid img_size in preprocessing_config.json")


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
    # 2. Detect checkpoint if mounted into the container
    # ---------------------------------------------------
    global ckpt_arch, ckpt_domain

    if MODEL_PATH is not None and MODEL_PATH.exists():
        ckpt_arch, ckpt_domain = parse_model_filename(MODEL_PATH)
    else:
        ckpt_arch, ckpt_domain = None, None

    # ---------------------------------------------------
    # 3. Adjust normalisation based on domain
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
    # 4. Build model architecture if a checkpoint is present
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
    # 5. Load weights and move model to device
    # ------------------------------------------
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # important for dropout/batchnorm

###########################################################
# 6. Health endpoint                                      #
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
# 7. Metadata endpoint                                    #
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
# 8. Prediction endpoint                                  #
###########################################################
@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Query(default="mobilenet_v2_cxr")):
    """
    Predict chest X-ray class from an uploaded image.

    This endpoint accepts:
    - file: The chest X-ray image (JPG or PNG format)
    - model: The model name to use for prediction (queary paramenter)
           Examples: ?model=mobilenet_v2_cxr, ?model=efficientnet_b0_imagenet
           If not specified, defaults to mobilenetv2_cxr (the best performer)

    Returns:
    - prediction_index: The numeric class index (0, 1, or 2)
    - prediction_class: The class name (Pneumonia, TB, or Normal)
    - model_name: Which model was used for this preduction
    - architecture: The neural network architecure (mobilenet_v2, efficientnet_b0, etc.)
    - domain: Whether the model was trained on CXR or ImageNet data
    - device: Whether inference ran on CPU or GPU
    - class_map: The mapping of class names to indices
    """

    # ----------------------------------------------------
    # 1.  Load the requested model on-demand.
    # ----------------------------------------------------
    # This function reads models_config.json, finds the model, instantiates it,
    # loads the checkpoint weights, and returns both the model and its metadata.
    try:
        loaded_model, model_metadata = load_model_from_config(model, device)
    except (ValueError, FileNotFoundError) as e:
        # If model doesn't exist or checkpont file is missing, return error
        return {"error": str(e), "status": 400} # A Status 400 is and HTTP error
                                                # code meaning "Bad Request".
                                                # This is a client-side problem, like
                                                # malformed syntax or invalida data.

    # ----------------------------------------------------
    # 2. Load the image file from the upload.
    # ----------------------------------------------------
    # Convert to RGB ensures it's in the correct colour format (3 channels).
    img = Image.open(file.file).convert("RGB")
    
    # ---------------------------------------------------------------------
    # 3. Get the correct normalisation values based on the model's domain
    # ---------------------------------------------------------------------
    # The model was trained with specific mean and std values.
    # We must use the SAME values during inference or predictions will be wrong.
    domain = model_metadata["domain"]
    if domain == "cxr":
        # Model was trained on chests X-ray data with CXR-specfic statistics.
        mean = cfg["cxr_mean"]
        std = cfg["cxr_std"]
    else:
        # Moedl was trained on chests X-ray data with ImageNet statistics (standard pretrained weights)
        mean = cfg["imagenet_mean"]
        std = cfg["imagenet_std"]

    # ----------------------------------------------------------------
    # 4. Build the preprocessing transforms (resize, normalize, etc).
    # ----------------------------------------------------------------
    # These transforms MUST match what was used during training.
    # build_shared_transforms returns (train_transforms, eval_transforms)
    # Use eval_transforms (the second return value, after underscore)
    _, tfm = build_shared_transforms(
        img_size=img_size,
        mean=mean,
        std=std,
        transform_mode=cfg.get("transform_mode", "center-crop")
    )
    
    # ---------------------------------------
    # 5. Apply the transforms to the image.
    # ---------------------------------------
    # tfm(img) applies all preprocessing (resize, crop, normalize).
    # .unsqueeze(0) adds a batch dimension: shape becomes [1, 3, 256, 256]
    # (batch_size=1, channels=3, height=256, width=256)
    x = tfm(img).unsqueeze(0)

    # -----------------------------------------------------
    # 6. Move the image to the correct device (CPU or GPU).
    # -----------------------------------------------------
    # This ensures the tensor is on the same device as the model.
    x = x.to(device)

    # -----------------------------------------------------------------------
    # 7. Get the model type from environment variable (for metrics tracking).
    # -----------------------------------------------------------------------
    # This tells us if CPU or GPU is running.
    model_type = os.getenv("MODEL_TYPE", "cpu")

    # -----------------------------------------------------
    # 8. Run the forward pass (inference) and measure time
    # -----------------------------------------------------
    # INFERENCE_DURATION_SECONDS.labels(model_type=model_type).time()
    # is a Prometheus metric that records how long inference takes.
    # torch.no_grad() tells PyTorch not to compute gradients (this is not training).
    # logits are the raw output numbers from the model before converting to probabilities.
    with INFERENCE_DURATION_SECONDS.labels(model_type=model_type).time():
        with torch.no_grad():
            logits = loaded_model(x)

    # ---------------------------------------
    # 9. Convert logits to class prediction. 
    # ---------------------------------------
    # logits.argmax(dim=1) finds with output is highest (most confident).
    # .item() converts PyTorch tensor to Python number
    # int() converts to integer (0, 1, or 2).
    pred_idx = int(logits.argmax(dim=1).item())

    # Softmax over the logits so we can show per-class probabilities
    # and the model's confidence in its top prediction. This is just
    # for display in the UI; it does not change the prediction itself.
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    confidence = float(probs[pred_idx])

    # ----------------------------------------
    # 10. Convert numeric index to class name. 
    # ----------------------------------------
    # Create reverse mapping: {0: "Pneumonia", 1: "TB", 2: "Normal"}
    idx_to_class = {v: k for k, v in cfg["class_to_index"].items()}
    pred_class = idx_to_class[pred_idx]

    # -----------------------------------------------------
    # 11. Record this prediction for metrics (Prometheus).
    # -----------------------------------------------------
    # This increments a counter: total predictions made.
    PREDICTIONS_TOTAL.labels(model_type=model_type).inc()

    # ----------------------------------------------------
    # 12. Return the prediction result with all metadata.
    # ----------------------------------------------------
    # Client gets the class name, the index, which model was used, and more.
    return {
        "prediction_index": pred_idx,
        "prediction_class": pred_class,
        "confidence": confidence,
        "probabilities": {idx_to_class[i]: float(p) for i, p in enumerate(probs)},
        "model_name": model_metadata["model_name"],
        "architecture": model_metadata["architecture"],
        "domain": model_metadata["domain"],
        "device": str(device),
        "class_map": cfg["class_to_index"]
    }

    
#######################################
# 9. List available models endpoint.  #
#######################################
@app.get("/dashboard-metrics")
def dashboard_metrics():
    """
    Lightweight JSON endpoint feeding the live dashboard widget in the UI.

    Returns a small set of operationally meaningful numbers:
      - uptime_seconds: how long this process has been running
      - models_available: count of model checkpoints registered in config
      - total_predictions: cumulative count across all model types
      - avg_inference_latency_ms: average wall-clock inference time
      - memory_mb: resident set size of this process in megabytes

    This is intentionally a small custom endpoint rather than parsing
    the full Prometheus /metrics output client-side.
    """
    # Uptime
    uptime = time.time() - APP_START_TIME

    # Models available (read fresh so a config change is reflected without restart)
    try:
        cfg_path = Path("/app/config/models_config.json")
        cfg = json.loads(cfg_path.read_text())
        models_available = len(cfg.get("available_models", {}))
    except Exception:
        models_available = 0

    # Total predictions: sum the Prometheus Counter samples across labels
    total_predictions = 0
    try:
        for sample in PREDICTIONS_TOTAL.collect()[0].samples:
            if sample.name.endswith("_total"):
                total_predictions += int(sample.value)
    except Exception:
        pass

    # Average inference latency: sum / count from the Histogram
    avg_latency_ms = 0.0
    try:
        total_sum = 0.0
        total_count = 0
        for sample in INFERENCE_DURATION_SECONDS.collect()[0].samples:
            if sample.name.endswith("_sum"):
                total_sum += sample.value
            elif sample.name.endswith("_count"):
                total_count += sample.value
        if total_count > 0:
            avg_latency_ms = (total_sum / total_count) * 1000.0
    except Exception:
        pass

    # Resident memory in MB from /proc/self/status (Linux only; container is Linux)
    memory_mb = 0
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    # Line looks like: "VmRSS:    389664 kB"
                    memory_mb = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass

    return {
        "uptime_seconds": int(uptime),
        "models_available": models_available,
        "total_predictions": total_predictions,
        "avg_inference_latency_ms": round(avg_latency_ms, 1),
        "memory_mb": memory_mb,
    }


@app.get("/models")
def list_models():
    """
    List all available models from models_config.json.

    Returns:
        JSON with model names, descriptions, and training domains.
        Clients use this to choose which model to use for prediction.

    Example response:
    {
      "available_models": [
        {
          "name": "mobilenet_v2_cxr",
          "description": "Optimised for speed (lightweight, fast inference)",
          "training_domain": "CXR data"
        }
      ]
    }
    """

    # Load the models configuration file.
    # This file lists all available models and their metadata.
    config_path = Path("/app/config/models_config.json")
    config = json.loads(config_path.read_text())

    # Extract the available_model dictionary from config.
    available = config["available_models"]

    # Build a simplified response for clients.
    # We only show information that helps them choose a model.
    # We hide internal details like checkpoint paths
    models_list = []

    # Iterate through each model in the config.
    for model_name, model_info in available.items():
        # Determine a human-readable description based on architecture.
        # This tells clients why they might choose this model.
        architecture = model_info["architecture"]

        if architecture == "mobilenet_v2":
            description = "Optimised for speed (lightweight, fast inference)"
        elif architecture == "efficientnet_b0":
            description = "Balanced performance and accuracy"
        elif architecture == "resnet50":
            description = "High accuracy (larger, slower inference)"
        elif architecture == "victorio":
            description = "Custom lightweight CNN"
        else:
            description = "Unknown architecture"

        # Build a model entry with only client-facing information.
        model_entry = {
            "name": model_name,
            "description": description,
            "training_domain": model_info["domain"]
        }

        # Add this entry to the list.
        models_list.append(model_entry)

        # Return the list sorted alphabetically for consistency.
        # Sorted by model name so clients see a predictable order.
    return {
        "available_models": sorted(models_list, key=lambda x: x["name"])
    }

#######################################
# 10. Serve static demo UI.           #
#######################################
# Mount the /demo directory as a static file server.
# This serves the demo/index.html file and any assets.
# Users access it at the site root /
app.mount("/", StaticFiles(directory="/app/demo", html=True), name="demo")