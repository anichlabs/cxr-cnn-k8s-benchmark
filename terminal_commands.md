terminal_commands.md:

## Terminal Commands for Building and Testing the Inference Containers

This document lists all terminal commands required to build, validate and run the CPU and GPU inference containers for the chest X-ray classification system. These commands reproduce the deployment layer described in Notebook 06 and allow verification of model loading, API operation and inference behaviour under both execution modes.

These instructions assume the user is positioned in the project root directory and has Podman installed with the NVIDIA Container Toolkit configured for GPU access. All commands here are deterministic and can be executed on any compatible workstation to reproduce the results presented in the thesis.
# ------------------------------------------------------------------------------

**Build GPU Image:**
podman build --no-cache \
  -f deployment/api/Containerfile.gpu \
  -t cxr-gpu .

**Validate GPU Import:**
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  cxr-gpu \
  python3 -c "import app.main; print('GPU IMPORT OK')"

  
**Build CPU Image:**
podman build --no-cache \
  -f deployment/api/Containerfile.cpu \
  -t cxr-cpu .

**Validate CPU Import:**
podman run --rm \
  cxr-cpu \
  python3 -c "import app.main; print('CPU IMPORT OK')"

# ------------------------------------------------------------------------------
**INSIDE the project root directory run these commands to test the CPU and GPU inference containers:**
# ------------------------------------------------------------------------------

-**GPU INFERENCE:**

RUN GPU CONTAINER:

podman run --rm \
  -p 8000:8000 \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v $(pwd)/experiments/checkpoints/mobilenet_v2_imagenet_best.pt:/app/model/mobilenet_v2_imagenet_best.pt:ro \
  -v $(pwd)/experiments/pools/preprocessing_config.json:/app/config/preprocessing_config.json:ro \
  cxr-gpu


CHECK GPU CONTAINER HEALTH:
curl http://localhost:8000/health


PREDICT WITH GPU CONTAINER:

IMG="/lab/px/msc-sept-24-ft-cohort-capstone-chrisanich/data/tbx11k/imgs/tb/tb0003.png"

curl -X POST http://localhost:8000/predict \
  -H "accept: application/json" \
  -F "file=@${IMG}"

# ------------------------------------------------------------------------------

-**CPU INFERENCE:**

RUN CPU CONTAINER:

podman run --rm \
  -p 8001:8000 \
  --security-opt=label=disable \
  -v $(pwd)/experiments/checkpoints/mobilenet_v2_imagenet_best.pt:/app/model/mobilenet_v2_imagenet_best.pt:ro \
  -v $(pwd)/experiments/pools/preprocessing_config.json:/app/config/preprocessing_config.json:ro \
  cxr-cpu


CHECK CPU CONTAINER HEALTH:
curl http://localhost:8001/health


PREDICT WITH CPU CONTAINER:

IMG="/lab/px/msc-sept-24-ft-cohort-capstone-chrisanich/data/tbx11k/imgs/tb/tb0003.png"

curl -X POST http://localhost:8001/predict \
  -H "accept: application/json" \
  -F "file=@${IMG}"

